import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import csv
import colorama
import sys
sys.setrecursionlimit(10000)

# init the colorama module
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET

# initialize the set of links (unique links)
internal_urls = set()
# external_urls = set()

total_urls_visited = 0

blacklist_neg = [
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

blacklist_pos = [
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
            "/faqs"
            ]


def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_website_links(url):
    global internal_urls
    """
    Returns all URLs that is found on `url` in which it belongs to the same website
    """
    # all URLs of `url`
    urls = set()
    try:
    # domain name of the URL without the protocol
        domain_name = urlparse(url).netloc
        soup = BeautifulSoup(requests.get(url).content, "html.parser",from_encoding="iso-8859-1")
        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                # href empty tag
                continue
            # join the URL if it's relative (not absolute link)
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            # remove URL GET parameters, URL fragments, etc.
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not is_valid(href):
                # not a valid URL
                continue
            if href in internal_urls:
                # already in the set
                continue
            if domain_name not in href:
                # external link
                # if href not in external_urls:
                #     print(f"{GRAY}[!] External link: {href}{RESET}")
                #     external_urls.add(href)
                continue
            
            # if not any(e in parsed_href.path for e in blacklist):
            #     internal_urls.add(href)
            print(f"{GREEN}[*] Internal link: {href}{RESET}")
            internal_urls.add(href)
            urls.add(href)

    except:
        print("[+] Total Internal links:", len(internal_urls))
        internal_urls = {i for i in internal_urls if not any(e in i for e in blacklist_pos)}
        print("After: "+str(len(internal_urls)))
        with open(f"pos_web_exp.csv",  mode='w', encoding='utf-8') as f:
            for internal_link in internal_urls:
                f.write(internal_link.strip()+'\n')
        

    return urls


def crawl(url,max_urls):
    """
    Crawls a web page and extracts all links.
    You'll find all links in `external_urls` and `internal_urls` global set variables.
    params:
        max_urls (int): number of max urls to crawl, default is 30.
    """
    global total_urls_visited
    total_urls_visited += 1
    links = get_all_website_links(url)
    for link in links:
        if total_urls_visited > max_urls:
            break
        crawl(link,max_urls)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Link Extractor Tool with Python")
    # parser.add_argument("url", help="The URL to extract links from.")
    # parser.add_argument("-m", "--max-urls", help="Number of max URLs to crawl, default is 30.", default=30, type=int)
    
    # args = parser.parse_args()
    # url = args.url
    # max_urls = args.max_urls
    
    url1 = "https://www.positive.news/articles/"
    url = "https://www.crimeonline.com/"

    crawl(url1,10000)

    print("[+] Total Internal links:", len(internal_urls))
    # print("[+] Total External links:", len(external_urls))
    # print("[+] Total URLs:", len(external_urls) + len(internal_urls))

    # domain_name = urlparse(url).netloc

    internal_urls = {i for i in internal_urls if not any(e in i for e in blacklist_pos)}
    print("All After: "+str(len(internal_urls)))
    # save the internal links to a file
    with open(f"pos_web.csv",  mode='w', encoding='utf-8') as f:
        for internal_link in internal_urls:
            f.write(internal_link.strip()+'\n')

    # # save the external links to a file
    # with open(f"{domain_name}_external_links.txt", "w") as f:
    #     for external_link in external_urls:
    #         print(external_link.strip(), file=f)