import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bs4 as bs
from urllib.request import urlopen
import requests
from dateutil.relativedelta import relativedelta
import pdb
from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player
import re


def getPlayersDf():
    teams2022 = Teams(year='2022')
    logfile = open('collectStaticData.log', 'w')
    logfile.write('Logging CollectStaticData.py on log file collectStaticData.log\n')
    logfile.write("------------------------------\n\n")
    ids, names, heights, weights, roles, roles2, salaries, shoots, debuts, births = []

    for team in teams2022:
        teamDict = Roster(team.abbreviation, year='2022',
                          slim=True).players
        for key, value in teamDict.items():
            logfile.write('Getting data for player: {0} ({1})'.format(key, value))
            try:
                if key not in ids:
                    player_dict = getPlayerData(key)
                    ids.append(key)
                    names.append(value)
                    salaries.append(Player(key).salary)
                    heights.append(player_dict['height'])
                    weights.append(player_dict['weight'])
                    roles.append(player_dict['first_pos'])
                    roles2.append(player_dict['second_pos'])
                    shoots.append(player_dict['hand'])
                    births.append(player_dict['birth'])
                    debuts.append(player_dict['debut'])
                    logfile.write('\tSuccessfully appended all data for player: {0} ({1})'.format(key, value))
            except Exception as e:
                logfile.write('Failed to append all data for player: {0} ({1})'.format(key, value))
                logfile.write('\t{1}'.format(str(e)))

    df = pd.DataFrame()
    df['Name'] = names
    df['Id'] = ids
    df['height'] = heights
    df['weights'] = weights
    df['role'] = roles
    df['role_2'] = roles2
    df['salary'] = salaries
    df['shoots'] = shoots
    df['debut'] = debuts
    df['birth'] = births

    logfile.close()
    return df


def getPlayerData(playerID):
    '''
    Gets the static data about a player by scraping
    it from https://www.basketball-reference.com.

    Parameters
    ----------
    The player ID to look for in basketball-reference.com

    Returns
    -------
    dictionary containing 'height', 'weight', 'first_pos', 'second_pos',
    'hand', 'birth', 'debut' statistics of the player
    '''

    url = f"https://www.basketball-reference.com/players/{playerID[0:1].lower()}/{playerID}.html"
    stats = None
    try:
        # print(url)
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        stats = [p.getText() for p in soup.find('div', {'id': 'meta'}).findAll('p')]
    except:
        raise Exception('Player {0} not found on basketball-reference.com'.format(playerID))
    statsDict = {
        'weight': None,
        'height': None,
        'hand': None,
        'first_pos': None,
        'second_pos': None,
        'birth': None,
        'debut': None
    }
    for stat in stats:
        stat = re.sub('[^a-zA-Z0-9]+', ' ', stat)
        try:
            if ("Position" in stat and "Shoots" in stat):
                statsDict['hand'] = stat.split('Shoots')[1].strip()
                posarr = stat.split('Shoots')[0].replace('Position', '').split('and')
                statsDict['first_pos'] = posarr[0].strip()
                if len(posarr) == 2:
                    statsDict['second_pos'] = posarr[1].strip()
            elif ("Position" in stat):
                posarr = stat.replace('Position', '').split('and').trim()
                statsDict['first_pos'] = posarr[0].strip()
                if len(posarr) == 2:
                    statsDict['second_pos'] = posarr[1].strip()
            elif ("Shoots" in stat):
                statsDict['hand'] = stat.replace('Shoots', '').strip()
            elif ("Debut" in stat):
                statsDict['experience'] = stat.replace('NBA Debut', '').strip()
            elif ("Born" in stat):
                statsDict['birth'] = stat.split('in')[0].replace('Born', '').strip()
            elif ('cm' in stat and 'kg' in stat and 'lb' in stat):
                hw = stat.split('lb')[1]
                statsDict['height'] = hw.split('cm')[0].strip()
                statsDict['weight'] = hw.split('cm')[1].replace(',', '').replace('kg', '').strip()
            else:
                continue
        except:
            raise Exception('Unable to parse line \'{0}\' for player {1}'.format(stat, playerID))
    return tuple(statsDict.values())


print(getPlayerData('labissk01'))
