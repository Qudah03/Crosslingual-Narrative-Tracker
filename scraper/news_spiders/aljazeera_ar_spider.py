import scrapy

class AlJazeeraArabicSpider(scrapy.Spider):
    name = "aljazeera_ar_headlines"
    allowed_domains = ["aljazeera.net"]
    start_urls = ["https://www.aljazeera.net/aljazeerarss/ar/all.xml"]

    def parse(self, response):
        for item in response.css("item"):
            yield {
                "title": item.css("title::text").get(),
                "link": item.css("link::text").get(),
                "pubDate": item.css("pubDate::text").get(),
                "language": "ar",
                "source": "AlJazeera_AR"
            }
# To run this spider, use the command:
# scrapy runspider scraper/news_spiders/aljazeera_ar_spider.py -o aljazeera_ar_headlines.json