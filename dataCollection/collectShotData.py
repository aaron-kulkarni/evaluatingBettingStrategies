import pandas as pd
from urllib.request import urlopen
import re
from sportsipy.nba.boxscore import Boxscores
import bs4
import numpy as np
import datetime as dt

teamAbbreviations = {
    "Raptors": "TOR",
    "Celtics": "BOS",
    "Nets": "BRK",
    "76ers": "PHI",
    "Knicks": "NYK",
    "Cavaliers": "CLE",
    "Bulls": "CHI",
    "Bucks": "MIL",
    "Pacers": "IND",
    "Pistons": "DET",
    "Hawks": "ATL",
    "Wizards": "WAS",
    "Heat": "MIA",
    "Hornets": "CHO",
    "Magic": "ORL",
    "Blazers": "POR",
    "Thunder": "OKC",
    "Jazz": "UTA",
    "Nuggets": "DEN",
    "Timberwolves": "MIN",
    "Warriors": "GSW",
    "Clippers": "LAC",
    "Suns": "PHO",
    "Kings": "SAC",
    "Lakers": "LAL",
    "Rockets": "HOU",
    "Grizzlies": "MEM",
    "Spurs": "SAS",
    "Mavericks": "DAL",
    "Pelicans": "NOP"
}


def scrapeGameShots(gameId):
    """
    Scrapes the shot chart data from
    basketball-reference.com for a particular game id
    Parameters
    ----------
    gameid : the basketball-reference.com boxscore id of the game to be scraped
    Returns
    -------
    A list as such: {
        [homeTeamAbbr, [arrayOfDistanceValuesHome], [arrayOfResultValuesHome], awayTeamAbbr, [arrayOfDistanceValuesAway], [arrayOfResultValuesAway]]
        arrayOfResultValues -> 0 for miss, 1 for make
    }
    """

    if gameId is None or not re.match(r"^[\d]{9}[A-Z]{3}$", gameId):
        raise Exception("Issue with game ID")    

    url = 'https://www.basketball-reference.com/boxscores/shot-chart/{0}.html'.format(gameId)

    try:
        soup = bs4.BeautifulSoup(urlopen(url), features='lxml')
        teams = soup.title
        teamsList = teams.next.split(' ')
        if 'Trail' in teamsList:
            teamsList.remove('Trail')
        #teamsList originally formats as ["HTeam", "vs", "ATeam,"]
        homeTeam = teamAbbreviations[teamsList[0]]
        #gets away team without comma at the end
        awayTeam = teamAbbreviations[teamsList[2][:len(teamsList[2])-1]]

        distances = []
        results = []
        homeShots = soup.find(id = 'shots-{}'.format(homeTeam))
        for shot in homeShots:
            if (shot != "\n" and shot.name != 'img'):
                info = shot.attrs['tip'].split("<br>")[1].split(" ")
                distances.append(info[5])
                if (info[2] == 'made'):
                    results.append(1)
                else:
                    results.append(0)
                
        homeList = [homeTeam, distances, results]
        
        distances = []
        results = []
        awayShots = soup.find(id = 'shots-{}'.format(awayTeam))
        for shot in awayShots:
            if (shot != "\n" and shot.name != 'img'):
                info = shot.attrs['tip'].split("<br>")[1].split(" ")
                while (len(info) > 7): #if player has middle name or suffix(ex. marvin bagley III)
                    info = info[1:]
                distances.append(info[5])
                if (info[2] == 'made'):
                    results.append(1)
                else:
                    results.append(0)

        awayList = [awayTeam, distances, results]

        return homeList + awayList

        #print(soup)
    except Exception as e:
        print("Failed to add game: {0}".format(gameId))
        return

#scrapeGameShots('202012230CHI')
def getTeamGameShotsDataFrame(year):

    #get gameId list from preexisting csv

    gameIdList = pd.read_csv('data/eloData/team_elo_{}.csv'.format(year), index_col = 0).index

    df = pd.DataFrame(index = gameIdList, columns = ['homeAbbr', 'homeDistances', 'homeResults', 'awayAbbr', 'awayDistances', 'awayResults'])
    
    for id in gameIdList:
        df.loc[id] = np.asarray(scrapeGameShots(id), dtype=object)

    print("Year finished: " + str(year))
    return df

years = np.arange(2020, 2023)
for year in years:
    getTeamGameShotsDataFrame(year).to_csv('data/shotData/team_shot_data_{}.csv'.format(year))

