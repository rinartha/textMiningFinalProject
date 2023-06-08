from bs4 import BeautifulSoup
import requests


class getNews():
    def __init__(self, website="https://www.taiwannews.com.tw/"):
        self.website = website

    def getContent(self, soup, tagName, subTag, element, elementName, link):
        # Find the content
        try:
            text = ""
            result = soup.find(tagName,  {element:elementName})
            paragraphs = result.find_all(subTag)
            if link == "content":
                for paragraph in paragraphs:
                    text = text + paragraph.text.strip() #+ "\n"
            else:
                for paragraph in range (1):
                    text = text + paragraphs[paragraph].get("href")
            return text
        except:
            pass
    
    def getNews(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        return soup

    def findTaiwanNews(self):
        soup = self.getNews(self.website)
        link = self.getContent(soup, "header", "a", "class", "entry-header", "link")
        lastNews = self.website + link
        soup = self.getNews(lastNews)
        content = self.getContent(soup, "div", "p", "itemprop", "articleBody", "content")
        
        return content

    def findTaipeiNews(self, website):
        soup = self.getNews(website)
        link = self.getContent(soup, "div", "a", "class", "boxTitle boxText bigbox", "link")
        soup = self.getNews(link)
        content = self.getContent(soup, "div", "p", "class", "archives", "content")
        return content
