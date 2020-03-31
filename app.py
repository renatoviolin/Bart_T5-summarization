# import config
import torch
import flask
from flask import Flask, request, render_template
import json
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

BART_PATH = 'bart-large'
T5_PATH = 't5-base'
# BART_PATH = 'model/bart'
# T5_PATH = 'model/T5'

app = Flask(__name__)
bart_model = BartForConditionalGeneration.from_pretrained(BART_PATH)
bart_tokenizer = BartTokenizer.from_pretrained(BART_PATH)

t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bart_summarize(input_text, num_beams=4, num_words=50):
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=int(num_beams),
                                      no_repeat_ngram_size=3,
                                      length_penalty=2.0,
                                      min_length=30,
                                      max_length=int(num_words),
                                      early_stopping=True)
    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


def t5_summarize(input_text, num_beams=4, num_words=50):
    input_text = str(input_text).replace('\n', '')
    input_text = ' '.join(input_text.split())
    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt")
    summary_task = torch.tensor([[21603, 10]])
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1)
    summary_ids = t5_model.generate(input_tokenized,
                                    num_beams=int(num_beams),
                                    no_repeat_ngram_size=3,
                                    length_penalty=2.0,
                                    min_length=30,
                                    max_length=int(num_words),
                                    early_stopping=True)
    output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sentence = request.json['input_text']
        num_words = request.json['num_words']
        num_beams = request.json['num_beams']
        model = request.json['model']
        if sentence != '':
            if model.lower() == 'bart':
                output = bart_summarize(sentence, num_beams, num_words)
            else:
                output = t5_summarize(sentence, num_beams, num_words)
            response = {}
            response['response'] = {
                'summary': str(output),
                'model': model.lower()
            }
            return flask.jsonify(response)
        else:
            res = dict({'message': 'Empty input'})
            return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')
    except Exception as ex:
        res = dict({'message': str(ex)})
        print(res)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    bart_model.to(device)
    bart_model.eval()
    t5_model.to(device)
    t5_model.eval()
    app.run(debug=True, use_reloader=False)
