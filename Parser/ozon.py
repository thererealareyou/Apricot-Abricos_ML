import selenium.common.exceptions
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np

import sqlite3

import time

def init_webdriver():
    driver = webdriver.Chrome()
    stealth(driver,
            platform="Win32")
    return driver


def scrolldown(driver, deep):
    for _ in range(deep):
        driver.execute_script('window.scrollBy(0, 600)')
        time.sleep(0.3)

def parse_categories():
    hrefs = []
    driver = init_webdriver()
    driver.get("https://ozon.ru")
    time.sleep(5)
    try:
        element = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[1]/div[1]/div/header/div[1]/div/div[1]/div/button")
        element.click()
    except selenium.common.exceptions.NoSuchElementException:
        ...

    actions = ActionChains(driver)

    for i in range(1, 20):
        we = driver.find_element(By.XPATH, "/html/body/div[3]/div/div/div[1]/ul/li[" + str(i) + "]")
        actions.move_to_element(we).pause(1).perform()

        table = driver.find_element(By.XPATH, "/html/body/div[3]/div/div/div[4]")
        a_elems = table.find_elements(By.TAG_NAME, 'a')
        for a_elem in a_elems:
            hrefs.append(a_elem.get_attribute("href"))

    with open('categories.txt', 'w') as f:
        for link in hrefs:
            f.write(link+'\n')

    return hrefs


def update_product_w_additional_data(): # WIP
    conn = None
    try:
        conn = sqlite3.connect('products_ozon.db', timeout=10)
        cursor = conn.cursor()

        cursor.execute('''
                INSERT INTO products (id, name, price, category, rating, reviews, link)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (id, name, price, category, rating, reviews, link))

        conn.commit()
        print(f"Товар '{name}' успешно добавлен.")

    except sqlite3.IntegrityError:
        print(f'Ошибка: товар с ID {id} уже существует.')
    except sqlite3.OperationalError as e:
        print(f'Ошибка базы данных: {e}')
    finally:
        if conn:
            conn.close()



def insert_product(id, name, price, category, rating, reviews, link):
    conn = None
    try:
        conn = sqlite3.connect('products_ozon.db', timeout=10)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO products (id, name, price, category, rating, reviews, link)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (id, name, price, category, rating, reviews, link))

        conn.commit()
        print(f"Товар '{name}' успешно добавлен.")

    except sqlite3.IntegrityError:
        print(f'Ошибка: товар с ID {id} уже существует.')
    except sqlite3.OperationalError as e:
        print(f'Ошибка базы данных: {e}')
    finally:
        if conn:
            conn.close()


def add_trader():
    ...


def get_top_of_category():
    conn = sqlite3.connect('products_ozon.db')

    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id VARCHAR PRIMARY KEY,
        name TEXT NOT NULL,
        trader TEXT,
        counter_amount INTEGER,
        counter_avg_price INTEGER,
        counter_avg_reviews INTEGER,
        direct_counter TEXT,
        direct_counter_reviews INTEGER,
        price INTEGER NOT NULL,
        category TEXT NOT NULL,
        rating REAL NOT NULL,
        reviews INTEGER NOT NULL,
        link TEXT NOT NULL
    )
    ''')

    conn.commit()
    conn.close()

    with open('categories_2.txt', 'r') as f:
        categories = [row.strip() for row in f]

    driver = init_webdriver()
    for category in categories:
        driver.get(category)
        scrolldown(driver, 30)
        time.sleep(5)
        category_name = str(driver.find_element(By.CLASS_NAME, 'qb6015-a1').text)
        print(category_name)
        table = driver.find_element(By.ID, "contentScrollPaginator")
        goods = table.find_elements(By.CLASS_NAME, "mj9_25")

        for good in goods:
            try:
                price = good.find_element(By.CLASS_NAME, "tsHeadline500Medium")
                price = int(str(price.text).replace(" ", '').replace('₽', ''))

                name = good.find_element(By.CLASS_NAME, 'tsBody500Medium')
                name = str(name.text)

                rating = good.find_element(By.CLASS_NAME, 'tsBodyMBold')
                elems = rating.find_elements(By.CLASS_NAME, 'p6b18-a4')
                rating = elems[0].text
                rating = float(rating)

                reviews = elems[1].text
                reviews = reviews.replace(' ', '')
                reviews = reviews[:reviews.find('отзыв')]
                reviews = str(reviews)
                print(rating, reviews)

                link = good.find_element(By.CSS_SELECTOR, 'a')
                link = link.get_attribute('href')
                link = str(link)

                id = link[:link.rfind('/')]
                id = id[id.rfind('-')+1:]
                print(id)

                insert_product(id, name, price, category_name, rating, reviews, link)

            except selenium.common.exceptions.NoSuchElementException or ValueError:
                continue


def get_full_info():
    conn = sqlite3.connect('products_ozon.db')
    cursor = conn.cursor()
    cursor.execute("SELECT link FROM products")
    results = cursor.fetchall()

    conn.commit()
    conn.close()

    driver = init_webdriver()
    for row in results:
        print(row[0])
        id = row[0]
        id = id[:id.rfind('/')]
        id = id[id.rfind('-') + 1:]
        print(id)

        driver.get(row[0])
        time.sleep(5)
        scrolldown(driver, 1)
        time.sleep(1)
        scrolldown(driver, 1)

        try:
            trader_list = driver.find_element(By.ID, 'seller-list')
            children = trader_list.find_elements(By.XPATH, './*')
            if len(children) > 0:
                button = children[-1]
                button.click()
        except selenium.common.exceptions.NoSuchElementException:
            ...

        try:
            trader_list = driver.find_element(By.CLASS_NAME, 'k5u_28')
            trader_list = trader_list.text.split('\n')
            print(trader_list[1])
        except selenium.common.exceptions.NoSuchElementException:
            trader = None

        try:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'mp8_28'))
            )
        except selenium.common.exceptions.TimeoutException:
            ...

        try:
            all_traders = driver.find_elements(By.CLASS_NAME, 'mp8_28')
            print(len(all_traders))

        except selenium.common.exceptions.NoSuchElementException:
            all_traders = []

        time.sleep(2)
        scrolldown(driver, 2)

        max_reviews = 0
        direct_counter = ''

        average_reviews = 0
        sum_reviews = 0

        average_price = 0
        sum_price = 0
        weighted_price = 0
        for i, tr in enumerate(all_traders):
            text = str(tr.text).split('\n')
            print(text)
            counter_name = text[0]
            counter_reviews = 0
            counter_price = 0
            for txt in text:
                if 'тзыв' in txt:
                    counter_reviews = txt[:txt.find(' ')]
                    sum_reviews += int(counter_reviews)
                    if int(counter_reviews) > max_reviews:
                        max_reviews = int(counter_reviews)
                        direct_counter = counter_name

                if '₽' in txt and not '%' in txt:
                    counter_price = txt.replace(' ', '').replace('₽', '')
                    sum_price += int(counter_price)

            average_reviews = float(sum_reviews / len(all_traders))
            average_price = float(sum_price / len(all_traders))

        print(f'"{counter_name}", "{counter_price}", "{counter_reviews}", "{average_price}", "{average_reviews}", "{direct_counter}", "{max_reviews}"')


