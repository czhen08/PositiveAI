import torch
import flask
import json
import torch.nn as nn
import news
from flask import Flask
from flask import request
from model import BertBinaryClassifier
from pytorch_pretrained_bert import BertTokenizer
from flask_cors import CORS
import schedule
import time
from threading import Thread
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL = None
device = "cpu"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

@app.route("/")
def index():
    return "<h1>PositiveAI Backend!</h1>"

@app.route("/update-news")
def updateNews():
   news.write_to_db()
   outlets = ["BBC", "DailyMail", "Guardian", "Metro", "Mirror", "Reuters", "Sun", "Independent"]
   for outlet in outlets:
      articles = news.query_db(outlet)
      pos_articles = getPositiveArticles(articles)
      news.write_to_table(outlet, pos_articles)
   print("Update at: "+str(datetime.now()))
   return json.dumps("success")

@app.route("/news")
def getNews():
   outlet = request.args.get("outlet")
   response = news.query_db(outlet)
   return json.dumps(response)

@app.route("/positive-news")
def getPostiveNews():
   outlet = request.args.get("outlet")
   response = news.query_db(str(outlet)+"_Positive")
   return json.dumps(response)

@app.route("/predict")
def predict():
   article = str(request.args.get("article"))
   article_tokens = tokenizer.tokenize(article)
   pos_scores = []
   article_segs_len = len(article_tokens)//510 + 1

   for i in range(article_segs_len):
      seg_tokens = ['[CLS]'] + article_tokens[510*i:(510*(i+1)-1)] + ['[SEP]']
      pos_scores.append(sentence_prediction(seg_tokens))
   positive_prediction = "article length: " + str(len(article_tokens))+ " words \n"

   for i in range(len(pos_scores)):
      positive_prediction += "[seg " + str(i+1) + "]:  " + str(pos_scores[i]) + "\n"
   response = {}
   response["response"] = {
      'sentence': str(article),
      'positive': str(positive_prediction)
   }
   return flask.jsonify(response)

def sentence_prediction(pred_tokens): 
   pred_tokens_id = tokenizer.convert_tokens_to_ids(pred_tokens)
   if len(pred_tokens_id)<512:
      pred_tokens_id += [0] * (512 - len(pred_tokens_id))
   predict_tokens_tensor = torch.tensor([pred_tokens_id]).to(device)
   pred = MODEL(predict_tokens_tensor)
   y_pred = pred.cpu().detach().numpy()[0][0]
   return round(y_pred, 3)

def getPositiveArticles(articles):
   pos_articles = []
   neg_articles = []
   for article in articles:
      article_tokens = ['[CLS]'] + tokenizer.tokenize(article['content'])[:510] + ['[SEP]']
      pos_score = sentence_prediction(article_tokens)
      if pos_score > 0.90:
         pos_articles.append(article)
      elif pos_score < 0.20:
         neg_articles.append(article)
   news.write_to_dataset(pos_articles,neg_articles)
   return pos_articles 

def updateDB():
   updateNews()
   schedule.every(30).minutes.do(updateNews)
   while True:
      schedule.run_pending()
      time.sleep(1)

t = Thread(target=updateDB)
t.start()

if __name__ == '__main__':
   MODEL = BertBinaryClassifier()
   # MODEL = nn.DataParallel(MODEL)
   MODEL.load_state_dict(torch.load("bert_model.bin", map_location=torch.device('cpu')))
   MODEL.to(device)
   MODEL.eval()
   app.run()