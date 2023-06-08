from flask import render_template, request, redirect
from finalProject import app
from finalProject.bot import *

ai = ChatBot(name="tommy")

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['text'] != "":
            datatext=request.form['text']
            allResult = ai.get_respond(datatext)

        if request.form['text'] != "" and request.form['textPrediction'] != "":
            textPrediction=request.form['textPrediction']
            predictionResult = ai.get_prediction(textPrediction)
            
            allResult = predictionResult + ':: none'

        return allResult

    return render_template('index.html')