import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from transformers import AutoTokenizer, TFAutoModel, TFRobertaModel, TFBertModel
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
tf.get_logger().setLevel(logging.ERROR)

#create class customBert as a modification of Bert
class customBert():
  #constructor function using bert tokenizer and bert model
  def __init__(self, tokenizer, bertModel):
    self.tokenizer = tokenizer
    self.bertModel = bertModel

  #function for tokenizing with data is a dataframe (train/test data)
  #maxlen is maximum length of text for tokenizing
  #text is the column name for text that will be processed
  def tokenizeProcess(self, data, maxlen, text):
    x_tok= self.tokenizer(
        text=data[text].tolist(),
        add_special_tokens=True,
        max_length=maxlen,
        truncation=True,
        padding="max_length", 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)
    return x_tok

  #creating a model with several layer with number of label/class/cluster with maximum length of sentence
  def create_model(self, numberCluster, max_len):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    #create embedding from bert model
    embeddings = self.bertModel(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = Dense(128, activation='relu', name='dense1')(out)
    out = tf.keras.layers.Dropout(0.1, name='dropout1')(out)
    out = Dense(32,activation = 'relu', name='dense2')(out)

    y = Dense(numberCluster,activation = 'softmax', name='dense3')(out)
        
    self.model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    return self.model
  
  #function for fine tune twice with new dataset based on previous model that has been trained with other dataset
  #numberCluster is a new number of cluster in a new dataset, with maxlen is the maximum length of a sentence
  def fineTune(self, numberCluster, max_len):
    
    for layer in self.model.layers:
      layer.trainable = True
    #set output of the previous model in one layer before classifier layer as an input for new layer
    #ignore the classifier layer from previous model (-1 is a classifier/output layer)
    #-2 means one layer before classifier layer
    #add some layers
    out = self.model.layers[-2].output
    out = Dense(128, activation='relu', name='dense4')(out)
    out = tf.keras.layers.Dropout(0.1, name='dropout2')(out)
    out = Dense(32,activation = 'relu', name='dense5')(out)
    y = Dense(numberCluster,activation = 'softmax')(out)
        
    self.model = tf.keras.Model(inputs=self.model.input, outputs=y)
    return self.model

  #compiling a model with specific learning rate
  #static loss function and metric
  def modelCompile(self, learningRate):
    optimizer = Adam(learning_rate=learningRate, 
                    epsilon=1e-08,
                    #decay=0.01,
                    clipnorm=1.0)

    # Set loss and metrics
    loss =  CategoricalCrossentropy()
    metric = CategoricalAccuracy('accuracy')
    self.model.compile(optimizer = optimizer,
                      loss = loss, 
                      metrics = metric)

  #fit the model with data result from bert tokenizer train, test with ids and attention mask
  #train_Y is label from training data
  #test_Y is label from test data
  def fitModel(self, x_train, x_test, Train_Y, Test_Y, epochs, batchSize):
    history = self.model.fit(x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
              y = to_categorical(Train_Y),
              validation_data = (
              {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, to_categorical(Test_Y)
              ),
            epochs=epochs,
              batch_size=batchSize
          )
    return history

  #save model in the form of h5 filetype in folderName
  def saveModel(self, fileNameModel):
    self.model.save(fileNameModel)

  #load model from folderName with h5 filetype TFAutoModel as model as it is used in bert model in this class
  def load_model_roberta(self, fileNameModel):
    self.model = load_model(fileNameModel, custom_objects={'TFRobertaModel':TFRobertaModel}, compile=True, options=None)

  #load model from folderName with h5 filetype TFBertModel as model as it is used in bert model in this class
  def load_model_bert(self, fileNameModel):
    self.model = load_model(fileNameModel, custom_objects={'TFBertModel':TFBertModel}, compile=True, options=None)

  #prediction function to predict test data into a label
  def pred(self, x_test):
    predicted_raw = self.model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
    result = np.argmax(predicted_raw, axis = 1)
    return result

  #prediction function to predict test data into a label
  def pred_single_text(self, max_len, testText):
    data_test = {"Text":[testText]}
    df_dataTest = pd.DataFrame(data_test)
    result = self.pred(self.tokenizeProcess(df_dataTest, max_len, 'Text'))
    return result

  #get the model that might be use to display a model summary
  def getModel(self):
    return self.model

# def main():
#   testText = """TAIPEI (Taiwan News) — Taiwan’s gross domestic product (GDP) per capita exceeded South Korea’s in 2022 
#             for the first time in a decade due to consistent higher average growth, the Ministry of Economic Affairs (MOEA) said Friday (April 28).
#             The average GDP per person in Taiwan reached US$32,811 (NT$1 million) in Taiwan, while South Korea recorded US$32,237 for last year, 
#             per CNA. The growth of the semiconductor industry and the return of Taiwanese investors from overseas helped Taiwan achieve an 
#             average yearly GDP growth of 3.2% over the past decade, while South Korea suffered under a declining currency to book only 2.6% growth per year.
#             The size of Taiwan’s manufacturing sector grew by a yearly average of 5.5% from 2013 to 2021, while in South Korea, manufacturing only 
#             expanded by an average of 2.8% per year during the same period. Exports also revealed a different pace of growth for the two countries, 
#             with an average annual growth rate of 4.6% for Taiwan and of 2.2% for South Korea, with the global average standing at 3%, according to MOEA data.
#             The gap in exports between the two has been narrowing over the past 10 years, as South Korea exported 1.8 times more than Taiwan in 2013, 
#             but 1.4 times more in 2021, the Economic Daily News reported."""

#   model_name = 'roberta-base'

#   dictLabel = {"Politics": 0, "Society": 1, "Business": 2}
#   tokenizerBert = AutoTokenizer.from_pretrained(model_name)
#   modelBert = TFAutoModel.from_pretrained(model_name)
#   resultBert = customBert(tokenizerBert, modelBert)
#   fileNameModel = 'model/robertaDoubleFineTuned.h5'
#   resultBert.load_model_roberta(fileNameModel)
#   tokenize_result = resultBert.pred_single_text(max_len=256, testText=testText)

#   print (list(dictLabel.keys())[list(dictLabel.values()).index(tokenize_result[0])])

# main()