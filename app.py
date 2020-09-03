import flask
import json
import torch
import torch.nn as nn
import news_operations
import schedule
import connect_db
import time
from flask import Flask
from flask import request
from model import BertBinaryClassifier
from pytorch_pretrained_bert import BertTokenizer
from flask_cors import CORS
from threading import Thread
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
outlets = ["BBC", "DailyMail", "Guardian", "Metro", "Mirror", "Reuters", "Sun", "Independent"]

@app.route("/")
def index():
    return "<h1>HappyNews Backend!</h1>"

@app.route("/news")
def get_headlines_db():
   """
   Take the outlet parameter from the request.
   Query all the news headlines of the news outlet from the database.
   """
   outlet = request.args.get("outlet")
   response = connect_db.query_headlines(outlet)
   return json.dumps(response)

@app.route("/positive-news")
def get_pos_headlines_db():
   """
   Take the outlet parameter from the request.
   Query positive news headlines of the news outlet from the database.
   """
   outlet = request.args.get("outlet")
   response = connect_db.query_headlines(str(outlet)+"_Positive")
   return json.dumps(response)

@app.route("/upload-news")
def upload_news():
   """
   Take a news url parameter from the request. Parse the url and predict the sentiment of the news. 
   If it is positive, store it into the database. Return the prediction results.
   """
   url = str(request.args.get("url"))
   article,outlet = news_operations.analyse_news_article(url)
   article_tokens = tokenizer.tokenize(article['content'])
   score, _, _ = calculate_score(article_tokens)
   if score >= 0.80:
      sentiment = 'Positive'
      connect_db.upload_pos_news(article,outlet)
   elif score <= 0.20:
      sentiment = 'Negative'
   else:
      sentiment = 'Neutral'
   prediction = "Score: "+str(score)+"\nSentiment: " + sentiment
   response = {}
   response["response"] = {
      'article': article,
      'prediction': prediction
   }
   return json.dumps(response)

@app.route("/predict")
def predict():
   """
   Take the text parameter from the request.
   Return a response that contains the prediction results of the text.
   """
   article = str(request.args.get("article"))
   article_tokens = tokenizer.tokenize(article)
   article_segs_len = len(article_tokens)//510 + 1
   pos_scores = []
   score, score1, score2 = calculate_score(article_tokens)
   if article_segs_len == 1:
      pos_scores.append(score)
   else: 
      pos_scores.append(score1)
      pos_scores.append(score2)

   prediction = "score: " + str(score) + "\n"
   prediction += str(len(article_tokens)) + " tokens \n"

   for i in range(len(pos_scores)):
      prediction += "[subtext " + str(i+1) + "]:  " + str(pos_scores[i]) + "\n"
   response = {}
   response["response"] = {
      'sentence': str(article),
      'positive': str(prediction)
   }
   return flask.jsonify(response)

def text_prediction(pred_tokens):
   """
   Helper function that takes text tokens as the parameter.
   Return the sentiment score for the text.
   """ 
   pred_tokens_id = tokenizer.convert_tokens_to_ids(pred_tokens)
   if len(pred_tokens_id)<512:
      pred_tokens_id += [0] * (512 - len(pred_tokens_id))
   predict_tokens_tensor = torch.tensor([pred_tokens_id]).to(device)
   pred = MODEL(predict_tokens_tensor)
   y_pred = pred.cpu().detach().numpy()[0][0]
   return round(y_pred, 3)

def calculate_score(article_tokens):
   """
   Helper function that takes atricles tokens as the parameter.
   Calculate the overall sentiment score for the article.
   """ 
   article_segs_len = len(article_tokens)//510 + 1
   if article_segs_len == 1:
      seg_tokens = ['[CLS]'] + article_tokens[:510] + ['[SEP]']
      score = text_prediction(seg_tokens)
      return score, -1, -1
   else: 
      seg_tokens_1 = ['[CLS]'] + article_tokens[:510] + ['[SEP]']
      seg_tokens_2 = ['[CLS]'] + article_tokens[510:510*2] + ['[SEP]']
      score1 = text_prediction(seg_tokens_1)
      score2 = text_prediction(seg_tokens_2)
      score = round(0.8*score1+0.2*score2,3)
      return score, score1, score2

def get_pos_articles(articles):
   """
   Helper function that takes a list of atricle dictionaries.
   Return a list that contains the positive atricle dictionaries.
   """ 
   pos_articles = []
   neg_articles = []
   for article in articles:
      article_tokens = tokenizer.tokenize(article['content'])
      score, _, _ = calculate_score(article_tokens)
      if score >= 0.80:
         pos_articles.append(article)
      elif score <= 0.20:
         neg_articles.append(article)
   return pos_articles 

def store_analysed_data():
   """
   Function for data analyse. Query the news data for each news outlet stored in the database. 
   Analyse their sentiment and store the results back to the database.
   """ 
   for outlet in outlets:
      articles_data = connect_db.query_historical_news(outlet)
      articles = analyse_data_sentiment(articles_data)
      connect_db.store_analysed_data(outlet,articles)
   return json.dumps("success")

def analyse_data_sentiment(articles):
   """
   Helper Function for data analyse. Takes queried news articles data, analyse their sentiment.
   Refine the data according to the database schema and returen the list of refined articles with sentiment.
   """
   analysed_articles = []
   for article in articles:
      article_tokens = tokenizer.tokenize(article['content'])
      score, _, _ = calculate_score(article_tokens)
      analysed_article = {'title': article['title'], 'content': article['content'], 'sentiment': score, 'published_at': article['published_at'], 'url': article['url']} 
      analysed_articles.append(analysed_article)
   return analysed_articles 

def update_news():
   """
   Collect and store real-time news headlines.
   Find out the positive ones and store them.
   """ 
   news_operations.store_headlines()
   for outlet in outlets:
      articles = connect_db.query_headlines(outlet)
      pos_articles = get_pos_articles(articles)
      connect_db.store_pos_headlines(outlet, pos_articles)
   return json.dumps("success")

def update_db():
   """
   Set up a schedule to update the headlines in the database every 30 minutes
   """ 
   update_news()
   schedule.every(30).minutes.do(update_news)
   while True:
      schedule.run_pending()
      time.sleep(1)

if __name__ == '__main__':
   MODEL = BertBinaryClassifier()
   MODEL.load_state_dict(torch.load("bert_model.bin", map_location=torch.device('cpu')))
   MODEL.to(device)
   MODEL.eval()
   app.run()

   # Start the thread to collect and analyse real-time headlines
   # update_news()
   # t = Thread(target=update_db)
   # t.start()