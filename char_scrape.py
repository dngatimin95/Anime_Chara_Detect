import os
import cv2
import sys
import io
import time
import requests
import hashlib
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager

def find_img(char_name, num_of_img):
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.set_window_size(1120, 1000)

    url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q="+character+"&oq="+character+"&gs_l=img"
    driver.get(url)

    def scroll_down(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    img_urls = set()
    img_count = 0
    results_start = 0
    while img_count < num_of_img:
        scroll_down(driver)

        thumbnail_results = driver.find_elements_by_css_selector("img.Q4LuWd")
        num_results = len(thumbnail_results)

        print(f"Found: {num_results} search results. Extracting links from {results_start}:{num_results}")

        for img in thumbnail_results[results_start:num_results]:
            try:
                img.click()
                time.sleep(1)
            except Exception:
                continue

            actual_img = driver.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_img:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    img_urls.add(actual_image.get_attribute('src'))

            img_count = len(img_urls)

            if len(img_urls) >= num_of_img:
                print(f"Found: {len(img_urls)} image links, done!")
                break
        else:
            print("Found:", len(img_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = driver.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                driver.execute_script("document.querySelector('.mye4qd').click();")

        results_start = len(thumbnail_results)
    return img_urls

def start():
    char_name = input("Enter the search term/character you want to find: ")
    file_path = input("Enter the file path where you want it saved: ")
    num_of_img = input("Enter the number of images you want to scrape: ")
    scrape(char_name.strip().lower(), file_path, int(num_of_img))

def scrape(char_name, file_path, num_of_img=10):
    char_folder = os.path.join(file_path,'_'.join(char_name.lower().split(' ')))
    if not os.path.exists(char_folder):
        os.makedirs(char_folder)

    links = find_img(char_name, num_of_img)
    for link in links:
        dl_image(char_folder, link)

def dl_image(folder_path, url):
    try:
        img_detail = requests.get(url).content
    except Exception as err:
        print(f"ERROR - Could not download {url} - {err}")

    try:
        img_file = io.BytesIO(img_detail)
        image = Image.open(img_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(img_detail).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as err:
        print(f"ERROR - Could not save {url} - {err}")

start()
