import scrapy

class DWSpider(scrapy.Spider):
    name = "dw_headlines"
    allowed_domains = ["dw.com"]
    start_urls = ["https://rss.dw.com/rdf/rss-en-all"]

    def parse(self, response):
        for item in response.xpath('//*[local-name()="item"]'):
            yield {
                'title': item.xpath('*[local-name()="title"]/text()').get(),
                'link': item.xpath('*[local-name()="link"]/text()').get(),
                'pubDate': item.xpath('*[local-name()="pubDate"]/text()').get(),
                'description': item.xpath('*[local-name()="description"]/text()').get(),
            }

# To run this spider, use the command:
# scrapy runspider scraper/news_spiders/dw_spider.py -o dw_headlines.json