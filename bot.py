import datetime
import numpy as np
import transformers

from transformers import AutoTokenizer, TFAutoModel
from finalProject.customBert import *
from finalProject.summarization import *
from finalProject.getWeather import *
from finalProject.getNewscontent import *

# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name
        model_name = "microsoft/DialoGPT-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.chatbot = transformers.pipeline("conversational", tokenizer = tokenizer, model=model_name)

        fileNameModel = "model/robertaDoubleFineTuned.h5"
        model_name = 'roberta-base'
        tokenizerBert = AutoTokenizer.from_pretrained(model_name)
        modelBert = TFAutoModel.from_pretrained(model_name)
        self.resultBert = customBert(tokenizerBert, modelBert)
        self.resultBert.load_model_roberta(fileNameModel)
        self.summarization = summary()
        self.weatherToday = weather()
        
    def get_respond(self, text):
        prediction = 'none'
        if self.name in text.lower():
            respond = "Hello I am "+self.name+" the AI (text to predict and summary). Instruction for me."
        
        elif "time" in text.lower():
            respond = datetime.datetime.now().time().strftime('%H:%M')

        elif "instruction" in text.lower():
            respond = "Time, Weather today, Prediction, News from Taiwan, News from Taipei"

        elif ("weather" in text.lower()) or ("weather" in text.lower() and "today" in text.lower()):
            respond = self.weatherToday.weather_today()
            
        elif "prediction" in text.lower():
            respond = "Please paste your text in the textarea, and say ready"
            prediction = 'display'

        elif "news" in text.lower() and "taiwan" in text.lower():
            dictLabel = {"Politics": 0, "Society": 1, "Business": 2}
            website = getNews()
            textContent = website.findTaiwanNews()
            
            tokenize_result = self.resultBert.pred_single_text(max_len=256, testText=textContent)

            predict = list(dictLabel.keys())[list(dictLabel.values()).index(tokenize_result[0])]
            summary = self.summarization.summary_result(textContent)
            respond = "News from taiwan news belongs to " + str(predict) + " topic, and the summary of this news is " + str(summary)
            prediction = textContent
        
        elif "news" in text and "taipei" in text:
            dictLabel = {"Politics": 0, "Society": 1, "Business": 2}
            website = getNews()
            textContent = website.findTaipeiNews("https://www.taipeitimes.com/")
            
            tokenize_result = self.resultBert.pred_single_text(max_len=256, testText=textContent)

            predict = list(dictLabel.keys())[list(dictLabel.values()).index(tokenize_result[0])]
            summary = self.summarization.summary_result(textContent)
            respond = "News from taipei times is belongs to " + str(predict) + " topic, and the summary of this news is " + str(summary)
            prediction = textContent
            
        elif any(i in text.lower() for i in ["thank","thanks"]):
            respond = np.random.choice(["You're welcome!","No problem!","Cool!","I'm here if you need me!","Peace out!"])

        else:
            # respond = "i don't get it"
            conversation = transformers.Conversation(text)
            conversation = self.chatbot(conversation, pad_token_id=50256)
            respond = conversation.generated_responses[-1]

        return respond + '::' + prediction

    def get_prediction(self, text):
        prediction = 'none'
        #predict = get prediction from model
        dictLabel = {"Politics": 0, "Society": 1, "Business": 2}
        tokenize_result = self.resultBert.pred_single_text(max_len=256, testText=text)

        predict = list(dictLabel.keys())[list(dictLabel.values()).index(tokenize_result[0])]
        summary = self.summarization.summary_result(text)
        return "This text is belongs to " + str(predict) + " topic, and the summary of this news is " + str(summary)