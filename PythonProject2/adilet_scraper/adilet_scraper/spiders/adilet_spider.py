import csv
import re
from pathlib import Path

import scrapy
from scrapy.exceptions import CloseSpider

from adilet_scraper.items import AdiletScraperItem


class AdiletSpider(scrapy.Spider):
    name = "adilet"
    allowed_domains = ["adilet.zan.kz"]

    def __init__(self, urls_file=None, limit=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls_file = Path(urls_file) if urls_file else None
        self.limit = int(limit) if limit else None
        self.output_path = Path(__file__).resolve().parents[2] / "output.csv"
        self.processed_urls = self._load_processed_urls(self.output_path)

    async def start(self):
        url_count = 0
        for url in self._iter_urls():
            if url in self.processed_urls:
                continue

            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.errback_http,
                dont_filter=True,
            )
            url_count += 1
            if self.limit and url_count >= self.limit:
                break

        if url_count == 0:
            raise CloseSpider("no_urls_to_crawl")

    def _iter_urls(self):
        candidate_paths = [
            self.urls_file,
            Path(__file__).resolve().parents[2] / "adilet_links.csv",
            Path(r"C:\Users\bulat\Desktop\copilot agent web\PythonProject2\adilet_links_BACKUP.csv"),
            Path(r"C:\Users\bulat\Desktop\copilot agent web\PythonProject2\adilet_document_links.csv"),
        ]
        candidate_paths = [path for path in candidate_paths if path]
        csv_path = next((path for path in candidate_paths if path.exists()), None)

        if not csv_path:
            self.logger.warning("Файл со ссылками не найден. Использую 1 тестовый URL.")
            yield "https://adilet.zan.kz/rus/docs/K1400000226"
            return

        self.logger.info("Загружаю ссылки из: %s", csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)

            if header and header[0].startswith("http"):
                yield header[0].strip()

            for row in reader:
                if not row:
                    continue
                url = row[0].strip()
                if url:
                    yield url

    def _load_processed_urls(self, output_path):
        if not output_path.exists():
            return set()

        processed = set()
        try:
            with output_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    url = (row or {}).get("url")
                    if url:
                        processed.add(url.strip())
        except Exception as exc:
            self.logger.warning("Не удалось прочитать output.csv для дедупликации: %s", exc)

        if processed:
            self.logger.info("Пропускаю уже собранные документы: %s", len(processed))
        return processed

    def errback_http(self, failure):
        self.logger.error("Ошибка загрузки %s: %s", failure.request.url, failure.value)

    def parse(self, response):
        if response.status != 200:
            self.logger.warning("Нестандартный статус %s для %s", response.status, response.url)
            return

        response_text_lower = response.text.lower()
        if "recaptcha" in response.url.lower() or "recaptcha" in response_text_lower:
            self.logger.warning("Сайт вернул капчу вместо документа: %s", response.url)
            return

        title = response.css("h1::text, title::text").get(default="").strip()
        if title.endswith(' - ИПС "Әділет"'):
            title = title.replace(' - ИПС "Әділет"', "").strip()

        content_selectors = [
            ".container_gamma.text.text_upd article *::text",
            ".container_gamma.text.text_upd *::text",
            "article *::text",
        ]
        content_parts = []
        for selector in content_selectors:
            content_parts = response.css(selector).getall()
            content_parts = [part.strip() for part in content_parts if part and part.strip()]
            if content_parts:
                break

        content = " ".join(content_parts)
        if not content:
            self.logger.warning("Не удалось извлечь текст документа: %s", response.url)
            return

        item = AdiletScraperItem()
        item["url"] = response.url
        item["doc_id"] = self._extract_doc_id(response.url)
        item["title"] = title or "Без заголовка"
        item["document_type"] = self._extract_document_type(title)
        item["status"] = self._extract_status(content)
        item["language"] = "rus" if "/rus/" in response.url else "kaz"
        item["authority"] = self._extract_authority(content)
        item["adoption_date"] = self._extract_adoption_date(content, title)
        item["content"] = content
        yield item

    @staticmethod
    def _extract_doc_id(url):
        match = re.search(r"/docs/([^/?#]+)", url)
        return match.group(1) if match else None

    @staticmethod
    def _extract_document_type(title):
        if not title:
            return None
        lowered = title.lower()
        mapping = [
            ("кодекс", "кодекс"),
            ("закон", "закон"),
            ("приказ", "приказ"),
            ("постановление", "постановление"),
            ("указ", "указ"),
            ("правила", "правила"),
            ("решение", "решение"),
        ]
        for needle, value in mapping:
            if needle in lowered:
                return value
        return "документ"

    @staticmethod
    def _extract_status(content):
        lowered = content.lower()
        if "утратил силу" in lowered:
            return "утратил силу"
        if "вводится в действие" in lowered:
            return "действует"
        return "не определен"

    @staticmethod
    def _extract_authority(content):
        lines = [line.strip() for line in content.split("  ") if line.strip()]
        patterns = [
            "Президент Республики Казахстан",
            "Правительство Республики Казахстан",
            "Министр",
            "Министерство",
            "Конституционный Суд",
            "Верховный Суд",
        ]
        for line in lines[:50]:
            for pattern in patterns:
                if pattern.lower() in line.lower():
                    return line
        return None

    @staticmethod
    def _extract_adoption_date(content, title):
        source = " ".join(filter(None, [title, content[:1000]]))
        match = re.search(
            r"от\s+(\d{1,2}\s+[а-яА-ЯёЁ]+\s+\d{4}\s+года)",
            source,
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else None
