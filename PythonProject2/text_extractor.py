import random
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def parse_single_document(url, session: requests.Session):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'ru-RU,ru;q=0.9'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find('h1')
        title = title_tag.text.strip() if title_tag else "Заголовок не найден"

        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        full_text = "\n".join(paragraphs)

        return {
            'url': url,
            'title': title,
            'content': full_text
        }
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")
        return None


if __name__ == "__main__":
    session = build_session()
    input_file = 'adilet_links_BACKUP.csv'
    output_file = 'adilet_parsed_texts_15k.csv'

    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден! Проверьте название.")
        exit()

    print("Загружаем базу ссылок...")
    df_links = pd.read_csv(input_file)

    # Берем последние 15 000 ссылок
    df_sample = df_links.tail(15000)
    urls_to_parse = df_sample['document_url'].tolist()

    # Система восстановления прогресса
    existing_urls = set()
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        existing_urls = set(df_existing['url'].tolist())
        print(f"Найдено {len(existing_urls)} уже скачанных документов в {output_file}.")
        print("Продолжаем работу с прерванного места...")

    target_size = len(urls_to_parse)

    for i, url in enumerate(urls_to_parse):
        # Если ссылка уже есть в итоговом файле, просто пропускаем ее
        if url in existing_urls:
            continue

        print(f"[{i + 1}/{target_size}] Сбор: {url}")
        doc_data = parse_single_document(url, session)

        if doc_data:
            # Превращаем один словарь в DataFrame
            df_single = pd.DataFrame([doc_data])
            # Если файла еще нет, пишем заголовки колонок, если есть - дописываем без заголовков
            header = not os.path.exists(output_file)
            # mode='a' (append) позволяет дописывать данные прямо на жесткий диск
            df_single.to_csv(output_file, mode='a', header=header, index=False, encoding='utf-8')

        # Пауза, чтобы сервер Әділет нас не заблокировал
        time.sleep(random.uniform(0.5, 1.5))

    print("\nСбор последних 15 000 документов успешно завершен!")
