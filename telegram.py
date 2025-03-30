import selenium.common.exceptions
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

import sqlite3
import time
import emoji
import pandas as pd


def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')


def remove_slashns(text):
    return text.replace('\n', ' ')


def init_webdriver():
    driver = webdriver.Chrome()
    stealth(driver,
            platform="Win32")
    return driver


def scrolldown(driver, deep):
    for _ in range(deep):
        driver.execute_script('window.scrollBy(0, 10)')
        time.sleep(0.3)


def get_info():
    driver = init_webdriver()
    types = ['20', '3']
    p_arrays = []
    themes = []

    for t in types:
        driver.get("https://tgstat.ru/tag/krasnodar-region")
        element = driver.find_element(By.ID, "categoryid")
        select = Select(element)
        time.sleep(1)
        element.click()
        time.sleep(1)
        select.select_by_value(t)
        time.sleep(1)

        groups = driver.find_elements(By.CLASS_NAME, 'card-body')
        links = []

        for group in groups:
            link = group.find_element(By.CLASS_NAME, 'text-body')
            link = link.get_attribute('href')
            links.append(str(link))

        for link in links:
            driver.get(link)
            time.sleep(2)

            posts = driver.find_elements(By.CLASS_NAME, 'post-text')
            for post in posts:
                p = post.text
                p = remove_emojis(p)
                p = remove_slashns(p)
                p_arrays.append(p)
                themes.append(t)
                print(p)

    df = pd.DataFrame({
        'Text': p_arrays,
        'Theme': themes
    })

    df.to_csv('data/data.csv', sep='|')


def get_specific_info(link):
    driver = init_webdriver()
    p_arrays = []

    driver.get("https://tgstat.ru/channel/" + link)
    time.sleep(2)

    posts = driver.find_elements(By.CLASS_NAME, 'post-text')
    for post in posts:
        p = post.text
        p = remove_emojis(p)
        p = remove_slashns(p)
        p_arrays.append(p)

    return p_arrays