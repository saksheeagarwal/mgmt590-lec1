import pandas as pd
from transformers.pipelines import pipeline
from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

@app.route("/answer", methods = ['POST'])
def answer():
    data = request.json
    hg_comp = pipeline('question-answering', model="bert-large-cased-whole-word-masking-finetuned-squad", tokenizer="bert-large-cased-whole-word-masking-finetuned-squad")
    answer = hg_comp({'question': data['question'], 'context': data['context']})['answer']
    return answer

if __name__ == '__main__' :
    app.run(host='0.0.0.0', port=8000, threaded=True)