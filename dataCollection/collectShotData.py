import pandas as pd
from urllib.request import urlopen
import re
from sportsipy.nba.boxscore import Boxscores
import bs4

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
    "Trailblazers": "POR",
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
    A dictionary as such: {
        'gameid' : 123456789ABC,
        'team abbr' : ABC,
        'distances' : [0, 0, 0, 0...]
        'result': [0, 0, 0, 0, 0...] (0 for miss, 1 for make)
    }
    """

    if gameId is None or not re.match(r"^[\d]{9}[A-Z]{3}$", gameId):
        raise Exception("Issue with game ID")    

    url = 'https://www.basketball-reference.com/boxscores/shot-chart/{0}.html'.format(gameId)

    try:
        soup = bs4.BeautifulSoup(urlopen(url), features='lxml')
        teams = soup.title
        teamsList = teams.next.split(' ')
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
                
        homeDict = {'gameId': gameId, 'team': homeTeam, 'distances': distances, 'result': results}
        
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

        awayDict = {'gameId': gameId, 'team': awayTeam, 'distances': distances, 'result': results}

        return homeDict, awayDict

        #print(soup)
    except Exception as e:
        print("Failed to add game: {0}".format(gameId))
        return
