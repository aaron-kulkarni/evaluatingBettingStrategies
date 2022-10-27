import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/Users/shaolin/iCloud/Project/scrapeOP-master')
from functions import *

import time

from selenium import webdriver
import pandas as pd


scrape_oddsportal_current_season(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', max_page = 4)

scrape_oddsportal_historical(sport = 'basketball', country = 'usa', league = 'nba', start_season = '2022-2023', nseasons = 1, current_season = 'yes', max_page = 2)


scrape_oddsportal_historical(sport = 'basketball', country = 'usa', league = 'nba',start_season = '2021-2022', nseasons = 1, current_season = 'no',start_page=6, max_page = 11)


scrape_oddsportal_historical(sport = 'basketball', country = 'usa', league = 'nba',start_season = '2015-2016', nseasons = 1, current_season = 'no',start_page=11, max_page = 16)

scrape_oddsportal_historical(sport = 'basketball', country = 'usa', league = 'nba',start_season = '2015-2016', nseasons = 1, current_season = 'no',start_page=16, max_page = 21)

scrape_oddsportal_historical(sport = 'basketball', country = 'usa', league = 'nba',start_season = '2015-2016', nseasons = 1, current_season = 'no',start_page=21, max_page = 26)

scrape_oddsportal_historical(sport = 'basketball', country = 'usa', league = 'nba',start_season = '2015-2016', nseasons = 1, current_season = 'no',start_page=26, max_page = 31)

df1 = pd.read_csv('/Users/shaolin/Library/Mobile Documents/com~apple~CloudDocs/Project/scrapeOP-master/nba/nba_2020-2021_1.csv', sep = ';')
df2 = pd.read_csv('/Users/shaolin/Library/Mobile Documents/com~apple~CloudDocs/Project/scrapeOP-master/nba/nba_2020-2021_5.csv', sep = ';')
df3 = pd.read_csv('/Users/shaolin/Library/Mobile Documents/com~apple~CloudDocs/Project/scrapeOP-master/nba/nba_2020-2021_11.csv', sep = ';')
df4 = pd.read_csv('/Users/shaolin/Library/Mobile Documents/com~apple~CloudDocs/Project/scrapeOP-master/nba/nba_2020-2021_16.csv', sep = ';')
df5 = pd.read_csv('/Users/shaolin/Library/Mobile Documents/com~apple~CloudDocs/Project/scrapeOP-master/nba/nba_2020-2021_21.csv', sep = ';')

df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
df.to_csv('closing_betting_odds_2021.csv',sep=';')

driver = webdriver.Chrome('/usr/local/bin/chromedriver')  # Optional argument, if not specified will search path.

driver.get('http://www.google.com/');

time.sleep(5) # Let the user actually see something!

search_box = driver.find_element_by_name('q')

search_box.send_keys('ChromeDriver')

search_box.submit()

time.sleep(5) # Let the user actually see something!

driver.quit()
