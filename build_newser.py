import newspaper
from newspaper import Article
import csv

texts = []
good_urls = []
with open('neg_news.csv') as fp:
  urls = fp.readlines()
print(len(urls))

for i in range(len(urls)):
  try:
    article = Article(urls[i].strip())
    article.download()
    article.parse()
    text = article.text

    text = text.replace('(Newser) – ', '')
    text = text.replace('story continues below', '')
    
    text = text.replace('\n','')
    texts.append(text)
    good_urls.append(urls[i])
    print(i)
  except:
    continue
print("-------------")

print(len(texts))
print(len(good_urls))

with open('neg_news_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'sentiment', 'url'])
    for i in range(len(good_urls)):
      writer.writerow([texts[i],'neg',good_urls[i]])

# writer = csv.writer(open('pos_web_dataset.csv', 'w', newline=''))
# writer.writerow(['text', 'sentiment', 'url'])
# for i in range(len(good_urls)):
#     writer.writerow([texts[i],'pos',good_urls[i]])