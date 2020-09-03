from __future__ import print_function
import time
import aylien_news_api
from aylien_news_api.rest import ApiException
from pprint import pprint
import connect_db
import news_parser

# Configure the AYLIEN News API
configuration = aylien_news_api.Configuration()
configuration.api_key['X-AYLIEN-NewsAPI-Application-ID'] = 'd366ed08'
configuration.api_key['X-AYLIEN-NewsAPI-Application-Key'] = 'dd725122b903b9e3a1b08c64daa95c8c'
configuration.host = "https://api.aylien.com/news"

# Create an instance of the API class
api_instance = aylien_news_api.DefaultApi(aylien_news_api.ApiClient(configuration))

# Definde the news outlets names and ids
outlets = ["BBC", "DailyMail", "Guardian", "Metro", "Mirror", "Reuters", "Independent", "Sun"]
ids = [29, 55, 6, 556, 1260, 27, 1078, 1551]

def get_headlines(outlet):
    """
    Take the news outlet name as the parameter. Use the News API to collect URLs of headlines.
    Use the news parser to parse the urls to get the headlines' information.
    Return a list of headlines.
    """
    if outlet == "BBC":
        parser = news_parser.BBC("https://www.bbc.co.uk")
    elif outlet == "DailyMail":
        parser = news_parser.DailyMail("https://www.dailymail.co.uk")
    elif outlet == "Guardian":
        parser = news_parser.Guardian("https://www.theguardian.com")
    elif outlet == "Metro":
        parser = news_parser.Metro("https://www.metro.co.uk")
    elif outlet == "Mirror":
        parser = news_parser.Mirror("https://www.mirror.co.uk/news/")
    elif outlet == "Reuters":
        parser = news_parser.Reuters("https://uk.reuters.com")
    elif outlet == "Sun":
        parser = news_parser.Sun("https://www.thesun.co.uk")
    elif outlet == "Independent":
        parser = news_parser.Independent("https://www.independent.co.uk")
    else:
        parser = news_parser.BBC("https://www.bbc.co.uk/news")
    
    index = outlets.index(outlet)
    url_list = []
    while len(url_list) < 50:
        opts = {
            'language': ['en'],
            'source_id': [ids[index]],
            'published_at_start':'NOW-1DAY',
            'published_at_end':'NOW',
            'sort_by': 'hotness',
            'sort_direction': 'desc',
            'cursor': '*',
            'per_page': 100
        }

        try:
            api_response = api_instance.list_stories(**opts)
            for story in api_response.stories:
                url = story.links.permalink
                if url:
                    url_list.append(url)
        except ApiException as e:
            print("Exception when calling DefaultApi->list_stories: %s\n" %e)
        
        opts['cursor'] = api_response.next_page_cursor
    
    url_list = url_list[:50]
    
    articles_list = []
    for url in url_list:
        raw_article = parser.get_article(url)
        if raw_article is not None:
          articles_list.append(raw_article)

    articles = []
    for article in articles_list:
        parsed_article = parser.parse(article)
        if parsed_article is not None:
          articles.append(parsed_article)
          
    if len(articles) > 30:
      articles = articles[:30]

    return articles

def analyse_news_article(url):
    """
    Take the url uploaded by the user in the frontend as the parameter.
    Use news parser to parse the url to get the news article's information.
    Return the parsed article.
    """
    if "bbc" in url:
        outlet = "BBC"
        parser = news_parser.BBC("https://www.bbc.co.uk/news")
    elif "dailymail" in url:
        outlet = "DailyMail"
        parser = news_parser.DailyMail("https://www.dailymail.co.uk")
    elif "theguardian" in url:
        outlet = "Guardian"
        parser = news_parser.Guardian("https://www.theguardian.com")
    elif "metro" in url:
        outlet = "Metro"
        parser = news_parser.Metro("https://www.metro.co.uk")
    elif "mirror" in url:
        outlet = "Mirror"
        parser = news_parser.Mirror("https://www.mirror.co.uk")
    elif "reuters" in url:
        outlet = "Reuters"
        parser = news_parser.Reuters("https://uk.reuters.com")
    elif "thesun" in url:
        outlet = "Sun"
        parser = news_parser.Sun("https://www.thesun.co.uk")
    elif "independent" in url:
        outlet = "Independent"
        parser = news_parser.Independent("https://www.independent.co.uk")
    else:
        outlet = "BBC"
        parser = news_parser.BBC("https://www.bbc.co.uk/news")
    
    article_data = parser.get_article(url)
    article = parser.parse(article_data)
    response = {'title': article['title'], 'content': article['content'], 'url': article['url'], 'image_url':article['metadata']['meta_img']} 
    return response, outlet


def store_headlines():
    """
    For each news outlet, collect its headlines and store the data into database.
    """
    for outlet in outlets:
        articles = get_headlines(outlet)
        connect_db.store_headlines(articles,outlet)

