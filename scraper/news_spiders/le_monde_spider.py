import scrapy

class LeMondeSpider(scrapy.Spider):
    name = "le_monde_headlines"
    allowed_domains = ["lemonde.fr"]
    start_urls = ["https://www.lemonde.fr/rss/en.xml"]  # English version for testing

    def parse(self, response):
        for item in response.css("item"):
            yield {
                "title": item.css("title::text").get(),
                "link": item.css("link::text").get(),
                "pubDate": item.css("pubDate::text").get()
            }
# To run this spider, use the command:
# scrapy runspider scraper/news_spiders/le_monde_spider.py -o le_m