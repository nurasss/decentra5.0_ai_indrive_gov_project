import random
import time
from urllib.parse import urljoin

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


def crawl_all_adilet_links(base_search_url):
    session = build_session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'ru-RU,ru;q=0.9'
    }

    all_document_links = set()
    base_domain = "https://adilet.zan.kz"
    page = 1  # Начинаем с первой страницы

    print("Запуск полного краулера. Идем до победного конца!")
    print("ВАЖНО: Для ручной остановки скрипта с сохранением данных нажмите Ctrl+C в терминале.")
    print("-" * 50)

    try:
        # Бесконечный цикл. Он прервется только командой break
        while True:
            params = {'page': page}
            print(f"Обработка страницы {page}...")

            try:
                response = session.get(base_search_url, headers=headers, params=params, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                links_found = 0
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if '/rus/docs/' in href and 'search' not in href:
                        full_url = urljoin(base_domain, href)
                        all_document_links.add(full_url)
                        links_found += 1

                print(f"  Найдено новых ссылок: {links_found}. Всего в базе: {len(all_document_links)}")

                # Условие выхода из бесконечного цикла
                if links_found == 0:
                    print("\nСсылки не найдены. Достигнут конец базы Әділет. Остановка цикла.")
                    break

                # АВТОСОХРАНЕНИЕ каждые 50 страниц (~1000 ссылок)
                if page % 50 == 0:
                    pd.DataFrame(list(all_document_links), columns=['document_url']).to_csv('adilet_links_BACKUP.csv',
                                                                                            index=False)
                    print(f"  [!] Сделан промежуточный бэкап (adilet_links_BACKUP.csv)")

            except Exception as e:
                print(f"  [Ошибка] на странице {page}: {e}")
                print("  Ждем 5 секунд и продолжаем со следующей...")
                time.sleep(5)  # При ошибке даем серверу "остыть" подольше

            page += 1  # Переходим к следующей странице
            time.sleep(random.uniform(1.0, 2.5))  # Стандартная пауза между запросами

    except KeyboardInterrupt:
        # Этот блок сработает, если вы нажмете Ctrl+C
        print("\n\n[ВНИМАНИЕ] Сбор прерван пользователем!")

    # ФИНАЛЬНОЕ СОХРАНЕНИЕ (сработает и при естественном конце, и при нажатии Ctrl+C)
    links_list = list(all_document_links)
    df_links = pd.DataFrame(links_list, columns=['document_url'])
    df_links.to_csv('adilet_document_links_FULL.csv', index=False)

    print("-" * 50)
    print(f"Сбор завершен. Итого уникальных ссылок: {len(links_list)}")
    print("Файл сохранен как 'adilet_document_links_FULL.csv'")
    return links_list


if __name__ == "__main__":
    search_url = "https://adilet.zan.kz/rus/search/docs/"
    # Запускаем нашу новую функцию без ограничений по страницам
    collected_links = crawl_all_adilet_links(search_url)
