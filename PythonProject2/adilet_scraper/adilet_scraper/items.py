import scrapy


class AdiletScraperItem(scrapy.Item):
    url = scrapy.Field()
    doc_id = scrapy.Field()
    title = scrapy.Field()
    document_type = scrapy.Field()
    status = scrapy.Field()
    language = scrapy.Field()
    authority = scrapy.Field()
    adoption_date = scrapy.Field()
    content = scrapy.Field()
