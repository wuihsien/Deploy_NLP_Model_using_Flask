from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from core import to_bert_ids, use_model,DataDic


import pickle
import pandas as pd
import os
import numpy as np
import torch
from transformers import BertTokenizer,BertConfig,BertForSequenceClassification
def answer(q_input):
    bert_ids = to_bert_ids(tokenizer,q_input)
    assert len(bert_ids) <= 512
    input_ids = torch.LongTensor(bert_ids).unsqueeze(0)

    # predict
    outputs = model(input_ids)
    predicts = outputs[:2]
    predicts = predicts[0]
    max_val = torch.max(predicts)
    label = (predicts == max_val).nonzero().numpy()[0][1]
    ans_label = answer_dic.to_text(label)
    return ans_label

# Preparing the Classifier
cur_dir = os.path.dirname(__file__)
data_features= pickle.load(open(os.path.join(cur_dir,
			'data_features.pkl'), 'rb'))
answer_dic = data_features['answer_dic']
        
# BERT
model_setting = {
        "model_name":"bert", 
        "config_file_path":"config.json", 
        "model_file_path":"pytorch_model.bin", 
        "vocab_file_path":"vocab.txt",
        #"vocab_file_path":"/home/ubuntu/Deploy_NLP_Model_using_Flask/vocab.txt",
        "num_labels":149  # 分幾類 
    }    

#model, tokenizer = use_model(**model_setting)
num_labels=149
tokenizer=BertTokenizer(vocab_file=os.path.join('vocab.txt'),do_lower_case=True)
config = BertConfig.from_pretrained(os.path.join('config.json'),num_labels=num_labels)
model = BertForSequenceClassification.from_pretrained('pytorch_model.bin',from_tf=True,config=config,force_download=True)
model.load_state_dict(torch.load(os.path.join('pytorch_model.bin')))
model.eval()

app = Flask(__name__)


@app.route('/')
def index():
	
	return render_template('index2.html')

@app.route('/results', methods=['GET'])
def predict():
	token= request.args.get('token')
	sOut = answer(token)
	return sOut

if __name__ == '__main__':
	app.run(debug=True)