############ Final oddsportal scraper

# ATP, baseball, basket, darts, eSports, football, nfl, nhl, rugby
''' Create 4 main functions : scrape_historical, scrape_specific_season, scrape current_season, scrape_next_games
NB : You need to be in the right repository to import functions...'''
import os

#os.chdir("C:\\Users\\Sébastien CARARO\\Desktop\\ATP& &Others\\WebScraping")
from functions import *

print('Data will be saved in the following directory:', os.getcwd())


scrape_oddsportal_historical(sport = 'basketball', country = 'usa', league = 'nba', start_season = '2020-2021', nseasons = 1, current_season = 'yes', max_page = 25)






