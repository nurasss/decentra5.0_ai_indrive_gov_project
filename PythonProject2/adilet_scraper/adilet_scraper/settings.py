BOT_NAME = "adilet_scraper"

SPIDER_MODULES = ["adilet_scraper.spiders"]
NEWSPIDER_MODULE = "adilet_scraper.spiders"

ROBOTSTXT_OBEY = False
COOKIES_ENABLED = False
TELNETCONSOLE_ENABLED = False

CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 8
DOWNLOAD_DELAY = 0.25
RANDOMIZE_DOWNLOAD_DELAY = True
DOWNLOAD_TIMEOUT = 45

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [403, 408, 429, 500, 502, 503, 504, 522, 524]

LOG_LEVEL = "INFO"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DOWNLOADER_CLIENT_TLS_VERIFY = False

ITEM_PIPELINES = {
    "adilet_scraper.pipelines.AdiletScraperPipeline": 300,
}

FEEDS = {
    "output.csv": {
        "format": "csv",
        "encoding": "utf-8-sig",
        "store_empty": False,
        "fields": [
            "url",
            "doc_id",
            "title",
            "document_type",
            "status",
            "language",
            "authority",
            "adoption_date",
            "content",
        ],
        "overwrite": False,
    },
}

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 5.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0

FEED_EXPORT_ENCODING = "utf-8"
