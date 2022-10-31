import sys
 
# adding Folder_2 to the system path
from functions import *

import time

from selenium import webdriver
import pandas as pd


scrape_oddsportal_current_season(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', max_page = 2)

scrape_oddsportal_next_games(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', nmax= 30)

