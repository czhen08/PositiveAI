import csv
import json
from newspaper import Article
import newspaper
from tqdm import tqdm
from connect_db import cursor,cnxn
import re
import pickle
import multiprocessing
from multiprocessing import Pool
import datetime
import os
import yaml
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import feedparser
from newspaper import Source
import unicodedata
import nltk

class Outlet:
    """
    Parent class for functions related to a generic newspaper.
    Newspapers without a specific class should default to this object.
    """

    def __init__(self, url):
        self.base_url = url
        # brand and container are initialised in Downloader()
        self.brand = ""
        self.container = None
        self.metadata_blacklist = { 'google-site-verification', 'msvalidate.01', 'og', 'videoUrl',  'scriptConfig', 'dataLayer', 'advertisingConfig', 'scriptConfig', 'dataLayer', 'p', 'manifest-validation', 'theme-color', 'publicationConfig', 'apple-itunes-app', 'versioned-resources', 'partnerConfig', 'partnerFooterConfig', 'referrer', 'advertisingConfig',  'manifest-validation', 'publicationConfig', 'apple-itunes-app', 'versioned-resources', 'partnerConfig', 'partnerFooterConfig', "fb", "viewport", "x-country", "x-audience", "CPS-AUDIENCE", "CPS_CHANGEQUEUEID", "image", "app_id", "admins", "apple-itunes-app", "apple-mobile-web-app-title", "application-name", "msapplication-TileImage", 'msapplication-TileColor', 'mobile-web-app-capable', 'robots', 'theme-color'}

    def get_frontpage_article_urls(self):
        """
        scrapes article urls from the front page. also attempts to scrape
        pagination urls and recursively scrapes articles from that too.
        """
        urls = list(self.get_urls_from_page(self.base_url))
        return urls

    def load_articles(self):
        """
        Loads list of articles already parsed and downloaded.
        """
        return None
    
    def get_article(self, url):
        """
        Takes as input an article url and:
        - downloads it
        - parses it
        - performs nlp

        returns the article object (part of newspaper3k)
        """
        article = Article(url)
        try:
            article.download()
            article.parse()
            article.nlp()
            return article
        except:
            return None

    def parse(self, article):
        """
        Takes as input some raw newspaper article object and
        returns a list of sentences/paragraphs representing
        the content of the article.
        """
        blacklist = {"clean_doc", "html", "article_html", "is_parsed", "movies", "text"}
        meta = {}
        base = {}
        
        base['date'] = self.get_article_date(article)

        for type in dir(article):
            # seperate metadata
            attribute = getattr(article,type)
            if "meta" in type:
                if type[0:3] == "set":
                    continue
                if type == "meta_data":
                    meta[type] = self.process_meta_data(attribute)
                    continue
                if type not in self.metadata_blacklist:
                    meta[type] = attribute
            else:
                if type in blacklist:
                    continue
                # if it's a method then skip it
                if callable(attribute):
                    continue
                if type[0:2] == "__":
                    continue
                if type[0:3] == "set":
                    continue
                if "img" in type or "image" in type:
                    continue
                if "download" in type:
                    continue
                if isinstance(attribute, set):
                    attribute = list(attribute)
                try:
                    test = json.dumps(attribute)
                    base[type] = attribute
                except:
                    continue
        base['metadata'] = meta

        # retrieve article content
        content = self.parse_article_content(article)
        # if there's nothing in the content then remove it.
        if len(content) < 1:
            return
        base['content'] = content

        return base

    def process_meta_data(self, metadata):
        """
        Filters contents in metadata
        """
        ents = {key:metadata[key] for key in metadata if key not in self.metadata_blacklist}
        # the really annoying tabloids (e.g. Mirror) have awkward nested meta_data
        if "meta_data" in ents:
            ents = {**ents, **self.process_meta_data(ents['meta_data'])}
        return ents

    def scrape_articles_in_urls(self, article):
        """
        Scrapes the website of the outlet for article URLs
        s.t. we can process them.
        """
        urls = set([i for i in article.doc.xpath('//@href')])
        urls = [i for i in urls if i[0:2] != "//"]
        urls = [i for i in urls if i[0:7] != "/assets"]
        urls = [i for i in urls if len(i) > 10]
        urls = [i for i in urls if (i[0] != "#") and (i != "/") and (i[0:4] != "/wp-")]
        urls = [i for i in urls if "xmlrpc" not in i]
        return urls

    def parse_article_content(self, article):
        """
        Takes the article object (part of newspaper3k) 
        and returns a list of paragraphs containing
        the content of the article.
        """
        try:
            children = article.clean_top_node.iterchildren()
            return ' '.join([x.text_content().strip() for x in children if x.tag == "p"])
            
        except:
            return ''
            
    @staticmethod
    def get_article_date(article):
        """
        Default mechanism for scraping the date from an article.
        (Some outlets will not store dates properly (e.g. BBC).)
        """
        if isinstance(article.publish_date, datetime.datetime):
            date = article.publish_date
        else:
            date = datetime.datetime.now()
        return date.strftime("%Y-%m-%d %H_%M")

    @staticmethod
    def make_permalink(url, root_url):
        """
        quick method to create permalink if necessary.
        (occasionally the websites would have relative
        links and not absolute links)
        """
        if url[0:4] != "http":
            return root_url + url
        return url

    @staticmethod
    def get_urls_from_page(url):
        """
        Recursively finds urls from a url. takes as input a url (string)
        and returns a set of urls
        """

        def get_next_url(page):
            """
            :param page: html of web page (here: Python home page) 
            :return: urls in that page 
            """
            start_link = page.find("a href")
            if start_link == -1:
                return None, 0
            start_quote = page.find('"', start_link)
            end_quote = page.find('"', start_quote + 1)
            url = page[start_quote + 1: end_quote]
            return url, end_quote

        # download raw html to start parsing.
        response = requests.get(url)
        page = str(BeautifulSoup(response.content,features="lxml"))

        urls = set()
        while True:
            url, n = get_next_url(page)
            page = page[n:]
            if url:
                urls.add(url)
            else:
                break
        return urls


class BBC(Outlet):
    def __init__(self, url):
        """
        It's hard to determine whether some article that was retrieved is
        garbage other than analysing the length of the article, so consider
        that instead.
        """
        super().__init__(url)
        self.exp = {
            'root' : 'https://www.bbc.co.uk',
            'article': "\/news/[A-z\-]+\d+$"
        }

    def get_frontpage_article_urls(self):
        frontpage = newspaper.build(self.base_url)
        return self.scrape_articles_in_urls(frontpage)

    def scrape_articles_in_urls(self, article):
        """
        Scrapes the article html for URLs.
        """
        urls = {i for i in article.doc.xpath('//@href')}
        urls = {i for i in urls if re.match(self.exp['article'], i)}
        urls = {self.make_permalink(i,self.exp['root']) for i in urls}
        return set(urls)

    @staticmethod
    def get_article_date(article):
        """
        Analyses the div tags to scrape the datetime.
        """
        try:
            content = article.clean_doc.xpath("//li[@class='mini-info-list__item']")[0]
            timestamp = int(content.iterchildren().__next__().values()[1])
            return datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H_%M")
        except:
            return datetime.datetime.now().strftime("%Y-%m-%d %H_%M")

class DailyMail(Outlet):
    def __init__(self, url):
        super().__init__(url)
        self.exp = {
            'root' : 'https://www.dailymail.co.uk',
            'article': "(\/[A-z0-9\_\-]+)+\/article-\d{7}\/[A-z0-9-]+\.html"
        }
        self.blacklist = {
            "/home/article-3633654/CONTRIBUTIONS-STANDARD-TERMS-CONDITIONS.html",
            "/home/article-1388040/Privacy-Policy-Cookies.html"
        }

    def get_frontpage_article_urls(self):
        urls = self.get_urls_from_page(self.base_url)
        urls = {i for i in urls if re.match(self.exp['article'], i)}
        urls = urls - self.blacklist
        urls = {self.make_permalink(i,self.exp['root']) for i in urls}
        return urls

    def scrape_articles_in_urls(self, article):
        """
        Scrapes the article html for URLs.
        """
        urls = set([i for i in article.doc.xpath('//@href')])
        urls = [i for i in urls if re.match(self.exp['article'], i)]
        urls = [self.make_permalink(i,self.exp['root']) for i in urls]
        return set(urls)

    def parse_article_content(self, article):
        children = article.clean_top_node.iterchildren()
        paragraphs = [x.text_content().strip() for x in children if x.tag == "p"]
        if len(paragraphs) < 2:
            return paragraphs
        if paragraphs[0] == paragraphs[1]:
            paragraphs = paragraphs[1:]
        if paragraphs[-1][0:4] == "MORE":
            paragraphs = paragraphs[:-1]
        return ' '.join(paragraphs)


class Guardian(Outlet):
    """
    Subclass representing parsing functions unique to The Guardian.
    -> https://www.theguardian.com
    """
    
    def __init__(self, url):
        super().__init__(url)

        self.exp = {
            'root' : 'https://www.theguardian.com',
            'article' : "\/[A-z0-9-]+\/\d+\/\w+\/\d+\/.+",
        }
        self.whitelist = set([])
        self.blacklist = set([])

    def get_frontpage_article_urls(self):
        """
        scrapes article urls from the front page. also attempts to scrape
        pagination urls and recursively scrapes articles from that too.
        """
        frontpage = newspaper.build(self.base_url)
        urls = set([i for i in frontpage.doc.xpath('//@href')])
        article_full_exp = self.exp['root'] + self.exp['article']
        good_urls = set([i for i in list(urls) if re.match(article_full_exp, i)])
        bad_urls = urls - good_urls
        return good_urls

    def scrape_articles_in_urls(self, article):
        """
        The guardian doesn't embed other articles in initial load,
        and does so with javascript.
        """
        return set()

    def parse_article_content(self, article):
        """
        Takes the article object (part of newspaper3k) 
        and returns a list of paragraphs.
        """
        children = article.clean_top_node.iterchildren()
        paragraphs = [x.text_content() for x in children if x.tag == "p"]
        return ' '.join(paragraphs)


class Metro(Outlet):
    def __init__(self, url):
        super().__init__(url)
        self.exp = {
            'root' : 'https://www.metro.co.uk',
            'article': "^(https:\/\/)?(metro\.co\.uk\/)?((\d+\/){3}).+\d/$"
        }

    def get_frontpage_article_urls(self):
        urls = self.get_urls_from_page(self.base_url)
        good_urls = [i for i in list(urls) if re.match(self.exp['article'], i)]
        urls = [self.make_permalink(i,self.exp['root']) for i in urls]
        good_urls = set(good_urls)
        bad_urls = set(urls) - good_urls
        return good_urls

    def scrape_articles_in_urls(self, article):
        """
        Scrapes the article html for URLs.
        """
        urls = set([i for i in article.doc.xpath('//@href')])
        urls = [i for i in urls if re.match(self.exp['article'], i)]
        urls = [self.make_permalink(i,self.exp['root']) for i in urls]
        return set(urls)

    def parse_article_content(self, article):
        children = article.clean_top_node.iterchildren()
        paragraphs = [x.text_content().strip() for x in children if x.tag == "p"]
        if len(paragraphs) < 1:
            return []
        if paragraphs[0] == paragraphs[1]:
            paragraphs = paragraphs[1:]
        paragraphs = [x for x in paragraphs if x[0:6] != "MORE: "]
        return ' '.join(paragraphs)


class Mirror(Outlet):
    """
    Subclass representing parsing functions unique to The Mirror.
    -> https://www.mirror.co.uk/
    """
    def __init__(self, url):
        super().__init__(url)

        self.exp = {
            'root' : 'https://www.mirror.co.uk',
            'article' : "\/([A-z0-9-]+\/){1,4}[A-z0-9-]+-\d{8}",
        }
        self.whitelist = set([])
        self.blacklist = set([])
        self.blackline = set(["The video will start in 8 Cancel", "Click to play Tap to play"])
    
    def get_frontpage_article_urls(self):
        """
        scrapes article urls from the front page. also attempts to scrape
        pagination urls and recursively scrapes articles from that too.
        """
        urls = self.get_urls_from_page(self.base_url)
        urls = {i for i in urls if i not in self.blacklist}
        article_full_exp = self.exp['root'] + self.exp['article']
        urls = {i for i in urls if re.match(article_full_exp, i)}
        urls = {self.filter_url(i) for i in urls}
        urls = {i for i in urls if "about-us" not in i}
        return urls

    def scrape_articles_in_urls(self, article):
        if article is None:
            return set()
        urls = {i for i in article.doc.xpath('//@href')}
        article_full_exp = self.exp['root'] + self.exp['article']
        urls = {i for i in urls if re.match(article_full_exp, i)}
        urls = {self.filter_url(i) for i in urls}
        urls = {i for i in urls if "about-us" not in i}
        return urls

    def parse_article_content(self, article):
        """
        Takes the article object (part of newspaper3k) 
        and returns a list of paragraphs.
        """
        if not article:
            return []
        if article.clean_top_node is None:
            return []
        children = article.clean_top_node.iterchildren()
        paragraphs = [x.text_content().strip() for x in children if x.tag == "p"]
        paragraphs = [i for i in paragraphs if i not in self.blackline]
        return ' '.join(paragraphs)

    @staticmethod
    def filter_url(url):
        comm = "#comments-section"
        if url[-17:] == comm:
            url = url[:-17]
        return url


class Reuters(Outlet):
    def __init__(self, url):
        super().__init__(url)
        self.exp = {
            'root' : 'https://uk.reuters.com',
            'article': '\/article\/[A-z0-9-]+\/[A-z0-9-]+id[A-Z0-9]{11}'
        }

    def get_frontpage_article_urls(self, base=None):
        """
        scrapes article urls from the front page. also attempts to scrape
        pagination urls and recursively scrapes articles from that too.
        """
        base = base if base else self.base_url
        urls = self.get_urls_from_page(base)
        urls = [i for i in urls if re.match(self.exp['article'], i)]
        urls = [self.make_permalink(i,self.exp['root']) for i in urls]
        return set(urls)

    def scrape_articles_in_urls(self,article):
        """
        Reuters does not set additional links in the articles.
        """
        return set()

    def pagination_scrape(self, pages=20):
        left = "https://uk.reuters.com/news/archive/domesticnews?view=page&page="
        right = "&pageSize=10"
        urls = set()
        for i in range(1,pages):
            num = str(i)
            url = left + num + right
            urls = urls.union(self.get_frontpage_article_urls(url))
        return urls


class Sun(Outlet):
    """
    Subclass representing parsing functions unique to The Sun.
    -> https://www.thesun.co.uk/
    """

    def __init__(self, url):
        super().__init__(url)

        self.exp = {
            'root' : 'https://www.thesun.co.uk',
            'article' : "/news/\d{7}/[A-z0-9-]+\/",
            'pagination' : "/([A-z\/]+)+page\/\d+\/",
        }
        self.whitelist = set([
            "https://www.thesun.co.uk/news/politics/",
            "https://www.thesun.co.uk/sport/",
            "https://www.thesun.co.uk/tech/",
            'https://www.thesun.co.uk/travel/',
            'https://www.thesun.co.uk/tvandshowbiz/',
            'https://www.thesun.co.uk/news/',
            'https://www.thesun.co.uk/motors/']
        )
        self.blacklist = set([
            "https://www.thesun.co.uk/video/"
        ])

    def get_frontpage_article_urls(self):
        """
        scrapes article urls from the front page. also attempts to scrape
        pagination urls and recursively scrapes articles from that too.

        Returns a set of URLs
        """
        
        feed = feedparser.parse(self.exp['root']+"/feed")
        urls = [entry.link for entry in feed.entries]
        urls = urls + list(self.get_urls_from_page(self.base_url))
        urls = [i for i in urls if i not in self.blacklist]
        article_full_exp = self.exp['root'] + self.exp['article']
        article_urls = set([i for i in urls if re.match(article_full_exp, i)])
        return article_urls

    def scrape_articles_in_urls(self, article):
        """
        Additional attempt to scrape more article URLs in 
        some article HTML.

        Returns a set of urls.
        """
        urls = list(set([i for i in article.doc.xpath('//@href')]))
        urls = [i for i in urls if re.match(self.exp['article'], i)]
        for x in range(len(urls)):
            url = urls[x]
            if  url[0:4].lower() != "http":
                urls[x] = self.exp['root']+url
        return set(urls)


class Times(Outlet):
    def __init__(self, url):
        super().__init__(url)

        self.exp = {
                'root' : 'https://www.thetimes.co.uk',
                'article' : "(https:\/\/www\.thetimes\.co\.uk)?\/edition\/[A-z0-9-]+\/[A-z0-9-_]+-[A-z0-9]{9}",
            }
        self.whitelist = set()
        self.blacklist = set()
        self.blackline = set()

        self.cookie_path = "../cookies/times_login.json"
        self.cookies = None
        
        with open("times.json") as json_file:
            payload = json.load(json_file)
            
#         self.s = requests.session()
#         self.s.post('https://login.thetimes.co.uk/', data=payload)

    def load_cookies(self):
        """
        Loads cookie json for the times (since the 
        articles are behind a paywall.)
        """
        with open(self.cookie_path) as file:
            self.cookies = json.load(item)
#         with open("times.json") as json_file:
#             payload = json.load(json_file)
            
#         self.s = requests.session()
#         self.s.post('https://login.thetimes.co.uk/', data=payload)
        pass
    
    def get_frontpage_article_urls(self):
        """
        scrapes article urls from the front page. also attempts to scrape
        pagination urls and recursively scrapes articles from that too.
        """
        home = newspaper.build(self.exp['root'])
        urls = set([i for i in home.doc.xpath('//@href')])
        urls = [i for i in urls if re.match(self.exp['article'], i)]
        urls = [self.make_permalink(i,self.exp['root']) for i in urls]
        return set(urls)

    def scrape_articles_in_urls(self, article):
        urls = set([i for i in article.doc.xpath('//@href')])
        urls = [i for i in urls if re.match(self.exp['article'], i)]
        urls = [self.make_permalink(i,self.exp['root']) for i in urls]
        return set(urls)
    
    def get_article(self, url):
        """
        Takes as input an article url and:
        - downloads it
        - parses it
        - performs nlp

        returns the article object (part of newspaper3k)
        """
        resp = requests.get(url)
        article = Article(url)
        try:
            article.set_html(resp.text)
            article.parse()
            article.nlp()
            return article
        except:
            return None

    def parse_article_content(self, article):
        """
        Takes the article object (part of newspaper3k) 
        and returns a list of paragraphs.
        """
        # parse article with beautiful soup
        parsed_html = BeautifulSoup(article.html, features="lxml")
        group = parsed_html.body.find('section', attrs={'class':'Article-body'})
        if group is not None:
            content = [p.text for p in group.find_all('p')][1:]
            bad = {"To save your favourite articles so you can find them later, subscribe to one of our packs."}
            paragraphs = [x for x in content if x not in bad]
            return ' '.join(paragraphs)
        else:
            return ''
        
    def parse(self, article):
        a = super().parse(article)
        # we tag this s.t we can use times_difixer.ipynb
        if a:
            a['verified'] = False
        return a
    
    
class Independent(Outlet):
    """
    Subclass representing parsing functions unique to The Independent.
    -> https://www.independent.co.uk/
    """

    def __init__(self, url):
        super().__init__(url)

        self.exp = {
            'root' : 'https://www.independent.co.uk',
            'article' : "https://www.independent.co.uk(\/[A-z0-9\-]+)+-a[0-9]{7}.html",
        }
        self.whitelist = set([
            "https://www.independent.co.uk/"
            "https://www.independent.co.uk/news/uk/",
            "https://www.independent.co.uk/news/world/",
            "https://www.independent.co.uk/news/world/americas/",
            'https://www.independent.co.uk/news/uk/politics/',
            'https://www.independent.co.uk/topic/brexit/',
            'https://www.independent.co.uk/final-say/',
            'https://www.independent.co.uk/news/science/',
            'https://www.independent.co.uk/environment/',
            'https://www.independent.co.uk/news/health/',
            'https://www.independent.co.uk/news/education/']
        )
        self.blacklist = set([
            "https://www.independent.co.uk/service/"
        ])
    
    def get_urls_from_page(self, source_url):
        boss = Source(source_url)
        boss.download()
        soup = BeautifulSoup(boss.html, 'lxml')
        # Extracting all the <a> tags into a list.
        tags = soup.find_all('a')
        # Extracting URLs from the attribute href in the <a> tags.
        links = [tag.get('href') for tag in tags]
        links = [x for x in links if x and (x[-4:] == "html")]
        for i in range(len(links)):
            url = links[i]
            if url[0] == "/":
                links[i] = source_url + url
        return set(links)
    
    def get_frontpage_article_urls(self):
        """
        scrapes article urls from the front page. also attempts to scrape
        pagination urls and recursively scrapes articles from that too.

        Returns a set of URLs
        """
        
        urls = []
        for base in self.whitelist:
            feed = feedparser.parse(base+"rss")
            urls = urls + [entry.link for entry in feed.entries]

        urls = urls + list(self.get_urls_from_page(self.base_url))
        # remove if blacklist url is found.
        for blacklist in self.blacklist:
            urls = [i for i in urls if blacklist not in i]
        article_urls = set([i for i in urls if re.match(self.exp['article'], i)])
        return article_urls

    def scrape_articles_in_urls(self, article):
        """
        Additional attempt to scrape more article URLs in 
        some article HTML.

        Returns a set of urls.
        """
        urls = list(set([i for i in article.doc.xpath('//@href')]))
        urls = [i for i in urls if re.match(self.exp['article'], i)]
        for x in range(len(urls)):
            url = urls[x]
            if  url[0:4].lower() != "http":
                urls[x] = self.exp['root']+url
        return set(urls)
    
    def parse_article_content(self, article):
        children = article.clean_top_node.iterchildren()
        paragraphs = [x.text_content().strip() for x in children if x.tag == "p"]
        paragraphs = [unicodedata.normalize('NFKD', x) for x in paragraphs if x]
        if paragraphs[-1] == "More follows...":
            paragraphs = paragraphs[:-1]
        return ' '.join(paragraphs)


def get_news(outlet):

    if outlet == "BBC":
        x = BBC("https://www.bbc.co.uk/news")
    elif outlet == "DailyMail":
        x = DailyMail("https://www.dailymail.co.uk")
    elif outlet == "Guardian":
        x = Guardian("https://www.theguardian.com")
    elif outlet == "Metro":
        x = Metro("https://www.metro.co.uk")
    elif outlet == "Mirror":
        x = Mirror("https://www.mirror.co.uk")
    elif outlet == "Reuters":
        x = Reuters("https://uk.reuters.com")
    elif outlet == "Sun":
        x = Sun("https://www.thesun.co.uk")
    elif outlet == "Independent":
        x = Independent("https://www.independent.co.uk")
    else:
        x = BBC("https://www.bbc.co.uk/news")

    url_set = x.get_frontpage_article_urls()
    url_list = list(url_set)[:30]
    articles_list = []
    for url in url_list:
        articles_list.append(x.get_article(url))

    articles = []
    for article in articles_list:
        articles.append(x.parse(article))

    # print(articles[5])
    # write_to_db(outlet,articles)
    return articles

def write_to_db():
    outlets = ["BBC", "DailyMail", "Guardian", "Metro", "Mirror", "Reuters", "Sun", "Independent"]
    for outlet in outlets:
        articles = get_news(outlet)
        db = outlet
        cursor.execute("delete from "+db)
        cnxn.commit()
        for article in articles:
            cursor.execute("insert into "+db+"(Title, Content, NewsURL, ImageURL) values (?,?,?,?)", str(article['title']), str(article['content']), str(article['url']), str(article['metadata']['meta_img']))
            cnxn.commit()
        print("Recorded: ",outlet)

def write_to_table(outlet, articles):
    cursor.execute("delete from "+outlet+"_Positive")
    cnxn.commit()
    for article in articles:
        cursor.execute("insert into "+outlet+"_Positive(Title, Content, NewsURL, ImageURL) values (?,?,?,?)", str(article['title']), str(article['content']), str(article['url']), str(article['image_url']))
        cnxn.commit()
    print("Recorded_pos: ",outlet)

def query_db(outlet):
    db = outlet
    cursor.execute("select Title, Content, NewsURL, ImageURL from "+db)
    rows = cursor.fetchall()
    articles = []
    for row in rows:
        article = {'title': row.Title, 'content': row.Content, 'url': row.NewsURL, 'image_url':row.ImageURL} 
        articles.append(article)
    return articles

def write_to_dataset(pos_articles,neg_articles):
    for article in pos_articles:
        cursor.execute("insert into training_dataset(Text, Sentiment, URL) values (?,?,?)", str(article['content']), 'pos', str(article['url']))
        cnxn.commit()

    for article in neg_articles:
        cursor.execute("insert into training_dataset(Text, Sentiment, URL) values (?,?,?)", str(article['content']), 'neg', str(article['url']))
        cnxn.commit()