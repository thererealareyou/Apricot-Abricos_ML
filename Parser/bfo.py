import selenium.common.exceptions
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By

import time

def init_webdriver():
    driver = webdriver.Chrome()
    stealth(driver,
            platform="Win32")
    return driver


def scrolldown(driver, deep):
    for _ in range(deep):
        driver.execute_script('window.scrollBy(0, 250)')
        time.sleep(0.3)


def get_info(inn):
    driver = init_webdriver()
    driver.get("https://bo.nalog.ru/search?query="+inn)
    time.sleep(3)
    try:
        result = driver.find_element(By.XPATH, '/html/body/div[1]/main/div/div/div[2]/div[2]/a')
    except selenium.common.exceptions.NoSuchElementException:
        return "По данному ИНН ничего не найдено."

    link = str(result.get_attribute('href'))
    print(link)

    driver.get(link)
    time.sleep(3)

    table = driver.find_element(By.XPATH, '/html/body/div[1]/main/div[2]/div[2]/div/div[2]/div[1]/div[2]/div/div/div')
    stickers = table.find_elements(By.CLASS_NAME, 'sticker')
    okved = None
    for sticker in stickers:
        if sticker.find_element(By.CLASS_NAME, 'sticker__title').text == 'Основной вид деятельности по ОКВЭД2':
            okved = sticker.find_element(By.CLASS_NAME, 'sticker__content').text
            okved = okved[:okved.find(' ')]

    if okved is None:
        return "На БФО не было найдено информации об ОКВЭД2"

    driver.get('https://star-pro.ru/proverka-kontragenta/organization')
    time.sleep(3)

    search = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/div/div[1]/div[2]/div[1]/div/div/div[1]/div/div[1]/div/button')
    time.sleep(1)
    search.click()

    elem = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/div/div[1]/div[2]/div[1]/div/div/div[3]/div/div/div/div[6]')
    time.sleep(1)
    elem.click()

    toggle = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/div/div[1]/div[2]/div[1]/div/div/div[3]/div/div/div/div[6]/div/div/div/div/div[1]/div[2]/label/span[1]/span[2]')
    time.sleep(1)
    driver.execute_script('window.scrollBy(-1000, 0)')
    time.sleep(1)
    toggle.click()

    search = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/div/div[1]/div[2]/div[1]/div/div/div[3]/div/div/div/div[6]/div/div/div/div/div[2]/div/div/div[2]/input')
    search.send_keys(okved)
    time.sleep(1)

    table = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/div/div[1]/div[2]/div[1]/div/div/div[3]/div/div/div/div[6]/div/div/div/div/div[4]/div/div')
    variants = table.find_elements(By.CSS_SELECTOR, 'span')

    variants[0].click()
    time.sleep(1)

    search = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/div/div[1]/div[2]/div[1]/div/div/div[3]/div/div/div/div[8]/button[2]')
    search.click()
    time.sleep(3)

    table = driver.find_elements(By.CLASS_NAME, 'OrganizationCard')
    links = []
    for elem in table:
        links.append(str(elem.find_element(By.CLASS_NAME, 'star-1nlh574').get_attribute('href')))

    for link in links:
        driver.get(link)
        time.sleep(2)
        name = driver.find_element(By.CSS_SELECTOR, 'h1').text
        print(name)
        data = driver.find_elements(By.CLASS_NAME, 'ftfbn-overall__stat')
        date = '-'
        for row in data:
            if "Регистрация" in row.text:
                date = str(row.find_element(By.CLASS_NAME, 'ftfbn-overall__value').text)
                date = date[:date.find(' ')]
                print(date)
                break

        desc = '-'
        desc = str(driver.find_element(By.CLASS_NAME, 'ftfbn-description__content').text)
        print(desc)
        print(link, '\n')