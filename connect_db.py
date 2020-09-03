import pyodbc
from datetime import datetime

# Database Information
server = 'positive-ai-server.database.windows.net'
database = 'positive-ai-db'
username = 'positive-ai-admin'
password = 'Zcdukic001'
driver= '{ODBC Driver 17 for SQL Server}'

def query_headlines(outlet):
    """
    Take the news outlet as a parameter.
    Query the headlines data stored in the database for the given news outlet.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    db = outlet
    cursor.execute("select Title, Content, NewsURL, ImageURL from "+db)
    rows = cursor.fetchall()
    articles = []
    for row in rows:
        article = {'title': row.Title, 'content': row.Content, 'url': row.NewsURL, 'image_url':row.ImageURL} 
        articles.append(article)
    return articles

def store_headlines(articles, db):
    """
    Take the news outlet name, scraped headlines as a parameters.
    Store the headlines into database.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute("delete from "+db)
    cnxn.commit()
    for article in articles:
        cursor.execute("insert into "+db+"(Title, Content, NewsURL, ImageURL) values (?,?,?,?)", str(article['title']), str(article['content']), str(article['url']), str(article['metadata']['meta_img']))
        cnxn.commit()

def store_pos_headlines(outlet, articles):
    """
    Take the news outlet name, positive headlines as a parameters.
    Store the positive headlines into database.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute("delete from "+outlet+"_Positive")
    cnxn.commit()
    for article in articles:
        cursor.execute("insert into "+outlet+"_Positive(Title, Content, NewsURL, ImageURL) values (?,?,?,?)", str(article['title']), str(article['content']), str(article['url']), str(article['image_url']))
        cnxn.commit()

def query_historical_news(outlet):
    """
    Take the news outlet as a parameter.
    Query the historical news data stored in the database, which is used for data analysis.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute("select NewsId, Title, Content, NewsUrl, Published_at from "+outlet+"_June")
    rows = cursor.fetchall()
    articles = []
    for row in rows:
        article = {'id':row.NewsId, 'title': row.Title, 'content': row.Content, 'published_at': row.Published_at, 'url': row.NewsUrl} 
        articles.append(article)
    return articles

def query_historical_news_sentiment(outlet):
    """
    Take the news outlet as a parameter.
    Query the sentiment score of the historical news data stored in the database, which is used for data analysis.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute("select NewsId, Sentiment, Published_at from "+outlet+"_June_Sentiment")
    rows = cursor.fetchall()
    articles_sentiment = []
    for row in rows:
        sentiment = float(str(row.Sentiment))
        articles_sentiment.append(sentiment)
    return articles_sentiment

def store_analysed_data(outlet, articles):
    """
    Take the news outlet as a parameter.
    Store the analysed news data their sentiment into the database.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute("delete from "+outlet+"_June_Sentiment")
    cnxn.commit()
    for article in articles:
        cursor.execute("insert into "+outlet+"_June_Sentiment(Title, Content, Sentiment, Published_at, NewsURL) values (?,?,?,?,?)", str(article['title']), str(article['content']), str(article['sentiment']), str(article['published_at']), str(article['url']))
        cnxn.commit()

def store_june_news(outlet, articles):
    """
    Take the news outlet and a list of articles as parameters.
    Store the api-collected news articles with certain publish time into database.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    for article in articles:
        cursor.execute("insert into "+outlet+"_June(Title, Content, NewsURL, Published_at, NewsSource) values (?,?,?,?,?)", str(article['title']), str(article['content']), str(article['url']), str(article['published_at']),str(article['source']))
        cnxn.commit()

def store_topic_news(outlet, articles):
    """
    Take the news outlet and a list of articles as parameters.
    Store the api-collected news articles with certain topics into database.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    for article in articles:
        cursor.execute("insert into "+outlet+"_Topics(Topic, Title, Content, URL) values (?,?,?,?)", str(article['topic']), str(article['title']), str(article['content']), str(article['url']))
        cnxn.commit()

def store_dataset(label,articles,db):
    """
    Store the news sentiment dataset into database
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    for article in articles:
        cursor.execute("insert into "+db+"(Text, Sentiment, URL) values (?,?,?)", str(article['content']), label, str(article['url']))
        cnxn.commit()

def query_dataset(db):
    """
    Query the news sentiment dataset on the database
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute("select text, sentiment from "+db)
    articles = []
    rows = cursor.fetchall()
    for row in rows:
        article = {'text': row.text, 'sentiment': row.sentiment} 
        articles.append(article)
    return articles

def upload_pos_news(article, outlet):
    """
    Take the news outlet name, and an postive article belongs to this outlet.
    Store the this positive article into database.
    """
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute("insert into "+outlet+"_Positive(Title, Content, NewsURL, ImageURL) values (?,?,?,?)", str(article['title']), str(article['content']), str(article['url']), str(article['image_url']))
    cnxn.commit()