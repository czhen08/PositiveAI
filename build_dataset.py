import newspaper
from newspaper import Article
import nltk
import csv

texts = []
good_urls = []
with open('good_news.csv') as fp:
  urls = fp.readlines()
for i in range(len(urls)):
  try:
    article = Article(urls[i])
    article.download()
    article.parse()
    text = article.text
    text = text.replace('\n','')
    # if(text != "Newsletter SignupDo you want to read more articles like this?"):
    texts.append(text)
    good_urls.append(urls[i])
    print(i)
  except:
    continue
print("-------------")
print(len(texts))
print(len(good_urls))

# with open(f"bad_news_dataset_test.csv",  mode='w', encoding='utf-8') as f:
#     f.write('text,sentiment,url\n')
#     for i in range(len(good_urls)):
#         f.write('"'+texts[i]+'",'+'neg,'+good_urls[i])

with open('good_news_dataset_local.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'sentiment', 'url'])
    for i in range(len(good_urls)):
      writer.writerow([texts[i],'pos',good_urls[i]])


# with open('bad_news_dataset.csv', encoding='utf-8') as fp:
#   lines = fp.readlines()
# print(lines[1])