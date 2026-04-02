class AdiletScraperPipeline:
    def process_item(self, item, *args, **kwargs):
        for field in ("title", "document_type", "status", "authority", "content"):
            value = item.get(field)
            if value:
                cleaned = str(value).replace("\r", " ").replace("\n", " ")
                item[field] = " ".join(cleaned.split())

        return item
