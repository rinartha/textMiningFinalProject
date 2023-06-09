# textMiningFinalProject

Tommy CAI
(Text tO predict and suMMarY Conversational Artificial Intelligence)
News Classification and Prediction Using Double Pre-trained Roberta Model With Text-to-speech Feature

Some files cannot be stored in this repository
1. cnn dataset (that can be found in https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning)
2. roberta pretrained model (you can use google colab file to create one of pretrained model and remember to use gpu environment)
3. virtual environment that I use in this project

This code consist of several parts/folder
1. dataset : cleaned dataset for training roberta base model. for cnn dataset you can download from the original source
2. google colab : file ipynb that I used for training roberta base model
3. model : location for the model (you need to create it first)
4. static : website asset that I use in this project
5. template : html file

Files
1. __init__.py
2. bot.py : class for AI bot 
3. customBert.py : class for training roberta base or bert base
4. getNewsContent.py : class for scraping data online by AI bot
5. getWeather.py : class for get the weather from google in realtime
5. route.py : main function
6. run.py
7. summarization.py : class for processing summary using bart model

To run this program in your local computer you need to install flask and some libraries that are used in this project. I use virtual environment to run this code.


This project is created by
group 18

Regards,

Komang Rinartha

