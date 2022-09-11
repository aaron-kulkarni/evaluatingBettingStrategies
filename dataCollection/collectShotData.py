import pandas as pd
from urllib.request import urlopen
import re
import bs4
import numpy as np
import math
import sys
sys.path.insert(0, "..")


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


scale = 1.2


def getAngle(top, left):

    # (56, 250) is the coordinate of the basket

    if (top - 56 > 0):
        return np.round(math.degrees(math.atan(abs(250-left)/(top-56))), 1)
    elif (250-left == 0):
        return 0
    else:
        return np.round(math.degrees(math.atan((56-top)/(abs(250-left)))), 1) + 90

def getDistance(top, left): #returns distance in INCHES!!!

    # (56, 250) is the coordinate of the basket
    return int(np.round((np.sqrt(((top-56) * (top-56)) + ((left-250) * (left-250)))) * scale))


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
        [homeTeamAbbr, [arrayOfDistanceValuesHome], [arrayOfResultValuesHome], [arrayOfShotAnglesHome], [arrayofPlayerIdsHome],
        awayTeamAbbr, [arrayOfDistanceValuesAway], [arrayOfResultValuesAway], [arrayofShotAnglesAway], [arrayofPlayerIdsAway]]

        (arrayOfResultValues -> 0 for miss, 1 for make)
    }
    """

    if gameId is None or not re.match(r"^[\d]{9}[A-Z]{3}$", gameId):
        raise Exception("Issue with game ID")    

    url = 'https://www.basketball-reference.com/boxscores/shot-chart/{0}.html'.format(gameId)

    try:
        soup = bs4.BeautifulSoup(urlopen(url), features='lxml')
        teams = soup.title
        teamsList = teams.next.split(' ')
        if 'Trail' in teamsList: #Portland's team name is "Trail Blazers" (two words), unlike all other single word team names
            teamsList.remove('Trail')
        #teamsList originally formats as ["HTeam", "vs", "ATeam,"]
        homeTeam = teamAbbreviations[teamsList[0]]
        #gets away team without comma at the end
        awayTeam = teamAbbreviations[teamsList[2][:len(teamsList[2])-1]]

        distances = []
        results = []
        angles = []
        threes = []
        playerids = []
        homeShots = soup.find(id = 'shots-{}'.format(homeTeam))
        for shot in homeShots:
            if (shot != "\n" and shot.name != 'img'):
                info = shot.attrs['tip'].split("<br>")[1]
                numbers = [int(s) for s in re.findall(r'-?\d+\.?\d*', info)]
                if (numbers[0] == 2):
                    threes.append(0)
                else:
                    threes.append(1)
                if ' missed ' in info:
                    results.append(0)
                else:
                    results.append(1)
                data = shot.attrs['style']
                res = [int(s) for s in re.findall(r'-?\d+\.?\d*', data)] #finds all ints in positions list
                angles.append(getAngle(res[0], res[1]))
                distances.append(getDistance(res[0], res[1]))
                data = shot.attrs['class']
                playerids.append(data[2][2:])
                
        homeList = [homeTeam, distances, results, angles, threes, playerids]
        
        distances = []
        results = []
        angles = []
        threes = []
        playerids = []
        awayShots = soup.find(id = 'shots-{}'.format(awayTeam))
        for shot in awayShots:
            if (shot != "\n" and shot.name != 'img'):
                info = shot.attrs['tip'].split("<br>")[1]
                numbers = [int(s) for s in re.findall(r'-?\d+\.?\d*', info)]
                if (numbers[0] == 2):
                    threes.append(0)
                else:
                    threes.append(1)
                if ' missed ' in info:
                    results.append(0)
                else:
                    results.append(1)
                data = shot.attrs['style']
                res = [int(s) for s in re.findall(r'-?\d+\.?\d*', data)] #finds all ints in positions list
                angles.append(getAngle(res[0], res[1]))
                distances.append(getDistance(res[0], res[1]))
                data = shot.attrs['class']
                playerids.append(data[2][2:])

        awayList = [awayTeam, distances, results, angles, threes, playerids]

        return homeList + awayList

    except Exception as e:
        print("Failed to add game: {0}".format(gameId))
        print(e)
        return

#print(scrapeGameShots('201411010WAS'))

def getTeamGameShotsDataFrame(year):

     #get game_id list from preexisting csv

     gameIdList = pd.read_csv('data/eloData/team_elo_{}.csv'.format(year), index_col = 0).index
     df = pd.DataFrame(index = gameIdList, columns = ['homeAbbr', 'homeDistances', 'homeResults', 'homeAngles', 'homeThrees', 'homePlayers', 'awayAbbr', 'awayDistances', 'awayResults', 'awayAngles', 'awayThrees', 'awayPlayers'])
    
     for id in gameIdList:
         df.loc[id] = np.asarray(scrapeGameShots(id), dtype=object)

     print("Year finished: " + str(year))
     return df
