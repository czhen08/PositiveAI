import newspaper
from newspaper import Article
import nltk
import csv


urls_positivenews = 'extracted_urls_positivenews.csv'
urls_goodnews = 'extracted_urls_goodnews.csv'
urls_crimeonline = 'extracted_urls_crimeonline.csv'

def parse_url(filename):
  """
  Open the CSV file contains URLs, parse the URLs using the newspaper module to get the body content.
  """
  articles = []
  valid_urls = []
  with open(filename) as fp:
    urls = fp.readlines()
  for url in urls:
    try:
      article = Article(url)
      article.download()
      article.parse()
      text = article.text
      text = text.replace('\n','')
      articles.append(text)
      valid_urls.append(url)
    except:
      continue
  return articles,valid_urls



def write_dataset_to_csv(filename, label, articles, valid_urls):
  """
  Write the data to a CSV file.
  """
  with open(filename, 'w', newline='', encoding='utf-8') as f:
      writer = csv.writer(f)
      writer.writerow(['text', 'sentiment', 'url'])
      for i in range(len(valid_urls)):
        writer.writerow([articles[i], label ,valid_urls[i]])


filename = 'good_news_dataset.csv'
label = 'pos'
articles, valid_urls = parse_url(urls_positivenews)
write_dataset_to_csv(filename, label, articles,valid_urls)