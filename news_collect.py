from __future__ import print_function
import time
import aylien_news_api
from aylien_news_api.rest import ApiException
from pprint import pprint
import connect_db
import collections
import pandas as pd

# Configure the AYLIEN News API
configuration = aylien_news_api.Configuration()
configuration.api_key['X-AYLIEN-NewsAPI-Application-ID'] = '9c6c27f8'
configuration.api_key['X-AYLIEN-NewsAPI-Application-Key'] = 'd92ec68d030138c25c386c1e0dec94c7'
configuration.host = "https://api.aylien.com/news"
api_instance = aylien_news_api.DefaultApi(aylien_news_api.ApiClient(configuration))

# Definde the news outlets names and ids
outlets = ["BBC", "DailyMail", "Guardian", "Metro", "Mirror", "Reuters", "Independent", "Sun"]
ids = [29, 55, 6, 556, 1260, 27, 1078, 1551]

def collect_news_articles():
    """
    Use the AYLIEN News API to collect news for each news outlet.
    Store the news articles into database.
    """
    for id in ids:
        for i in range(1, 31):
            if i < 10:
                start_date = '0'+str(i)
            else:
                start_date = str(i)
        
            if i < 9:
                end_date = '0'+str(i+1)
            else:
                end_date = str(i+1)

            published_at_start = '2020-06-' + start_date + 'T00:00:00Z'
            published_at_end = '2020-06-'+ end_date +'T00:00:00Z'
            if i == 30:
                published_at_end = '2020-07-01T00:00:00Z'

            opts = {
            'language': ['en'],
            'published_at_start': published_at_start,
            'published_at_end': published_at_end,
            'source_id': [id],
            'sort_by': 'hotness',
            'sort_direction': 'desc',
            'per_page': 100
            }
            try:
                api_response = api_instance.list_stories(**opts)
                articles = []
                url_set = set()

                for story in api_response.stories:
                    if story.links.permalink not in url_set:
                        url_set.add(story.links.permalink)
                        text = story.body
                        text = text.replace('\n','')
    
                        article = {'title': story.title, 'content': text, 'url': story.links.permalink, 'published_at':story.published_at, 'source': story.source.name} 
                        articles.append(article)
    
                articles = articles[:50]
                connect_db.store_june_news(outlets[ids.index(id)], articles)

            except ApiException as e:
                print("Exception when calling DefaultApi->list_stories: %s\n" % e)


