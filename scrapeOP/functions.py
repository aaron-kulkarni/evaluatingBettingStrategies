import sys
 
# adding Folder_2 to the system path
from functions import *

import time

from selenium import webdriver
import pandas as pd

scrape_oddsportal_current_season(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', max_page = 2)

driver = webdriver.Chrome('/usr/local/bin/chromedriver')  # Optional argument, if not specified will search path.

driver.get('http://www.google.com/');

time.sleep(5) # Let the user actually see something!

search_box = driver.find_element_by_name('q')

search_box.send_keys('ChromeDriver')

search_box.submit()

time.sleep(5) # Let the user actually see something!

driver.quit()
