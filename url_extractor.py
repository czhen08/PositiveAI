import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import csv
import colorama
import sys
sys.setrecursionlimit(10000)

colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET

# initialize the set of internal links (for unique links)
internal_urls = set()
total_urls_visited = 0

# set up the black list for each website
blacklist_crimeonline = [
            "/photos/pool",
            "/podcast/",
            "nancybook",
            "/nancy/",
            "/page",
            "/book/",
            "/author/",
            "/category/",
            "/privacy-policy/",
            "/terms-of-service/",
            "/about-crime-online/"]

blacklist_positivenews = [
            "/tag",
            "/category",
            "/support",
            "/articles",
            "/about",
            "/magazine",
            "/brands-of-inspiration",
            "/my-account",
            "/cart",
            "/shop",
            "/article-type",
            "/contact-us",
            "/privacy-policy",
            "/cookie-policy",
            "/partners",
            "/terms-and-conditions",
            "/rules",
            "/join",
            "/wp-content",
            "/subscribe",
            "/stock-positive-news-magazine",
            "/author",
            "/donate",
            "/faqs"]

blacklist_goodnews = [
            "/tag",
            "/category",
            "/members",
            "/author",
            "/contribute",
            "/more",
            "/cart",
            "/login",
            "/privacy-tools",
            "/wp-content",
            "/shop",
            "/product-tag",
            "/my-account",
            "/account",
            "/users",
            "/logout",
            "/privacy",
            "/app",
            "/contact",
            "/subscribe",
            "/gallery",
            ".html",
            "/news",
            "/users",
            "/admin",
            "/email-protection",
            "/rss",
            "/place_an_ad",
            "/opinion",
            "/our-privacy-policy",
            "/search",
            "/terms"]


def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_website_urls(url):
    """
    Returns all URLs that is found on the parameter `url` in which it belongs to the same website
    """
    global internal_urls
    urls = set()
    try:
        domain_name = urlparse(url).netloc
        soup = BeautifulSoup(requests.get(url).content, "html.parser",from_encoding="iso-8859-1")
        for a_tag in soup.findAll("a"):
            # Remove unnecessary information in the URLs. Get rid of all invalid URLs
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not is_valid(href):
                continue
            if href in internal_urls:
                continue
            if domain_name not in href:
                continue
            
            print(f"{GREEN}[*] Internal link: {href}{RESET}")
            internal_urls.add(href)
            urls.add(href)

    except:
        print("An Error occurred")
        return urls
        
    return urls


def scrape(url,max_urls):
    """
    Scrape a web page and extracts all links. Define max_urls as number of max urls to scrape
    Stop when the number of visited urls reaches max_urls.
    """
    global total_urls_visited
    total_urls_visited += 1
    links = get_website_urls(url)
    for link in links:
        if total_urls_visited > max_urls:
            break
        scrape(link,max_urls)


if __name__ == "__main__":
    # Define the domain urls of websites we want to scrape
    positive_news_url = "https://www.positive.news/articles/"
    crimeonline_url = "https://www.crimeonline.com/"
    good_news_url = "https://www.goodnewsnetwork.org/category/news/"

    # Scrape the website 
    scrape(positive_news_url,10000)

     # Save the extracted links to a file
    internal_urls = {i for i in internal_urls if not any(e in i for e in blacklist_positivenews)}
    with open(f"extracted_urls_positivenews.csv",  mode='w', encoding='utf-8') as f:
        for internal_link in internal_urls:
            f.write(internal_link.strip()+'\n')
