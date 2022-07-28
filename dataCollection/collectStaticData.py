import pandas as pd
import numpy as np
import bs4 as bs
from urllib.request import urlopen
from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player
import re


def getAllStaticPlayerData(year, logdir=''):
    """
    Gets static data for every player in a given year

    Parameters
    ----------
    year : the year to get all players for
    logdir :

    Returns
    -------
    A dataframe containing each player's static attributes in the given year
    """
    teams2022 = Teams(year=str(year))
    logfile = open('{0}collectStaticData{1}.log'.format(logdir, year), 'w', encoding='utf-8')
    logfile.write('Logging CollectStaticData.py on log file collectStaticData{0}.log\n'.format(year))
    logfile.write("------------------------------\n\n")
    ids, names, heights, weights, pgs, sgs, sfs, pfs, ces, salaries, shoots, debuts, experiences, births = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for team in teams2022:
        team_dict = Roster(team.abbreviation, year=str(year),
                           slim=True).players
        for key, value in team_dict.items():
            logfile.write('Getting data for player: {0} ({1})\n'.format(key, value))
            print('Getting data for player: {0} ({1})'.format(key, value))
            try:
                if key not in ids:
                    ids.append(key)
                    names.append(value)
                    try:
                        salaries.append(Player(key).salary)
                    except Exception:
                        logfile.write('\tFailed to append all data for player (salary): {0} ({1})\n'.format(key, value))
                        salaries.append(None)
                    player_dict = _getStaticPlayerData(key)
                    heights.append(player_dict['height'])
                    weights.append(player_dict['weight'])
                    pgs.append(player_dict['Point Guard'])
                    sgs.append(player_dict['Shooting Guard'])
                    sfs.append(player_dict['Small Forward'])
                    pfs.append(player_dict['Power Forward'])
                    ces.append(player_dict['Center'])
                    shoots.append(player_dict['hand'])
                    births.append(player_dict['birth'])
                    debuts.append(player_dict['debut'])
                    experiences.append(player_dict['experience'])
                    logfile.write('\tSuccessfully appended all data for player: {0} ({1})\n'.format(key, value))
                else:
                    logfile.write('\tFailed to append all data for player (duplicate): {0} ({1})\n'.format(key, value))
            except Exception as e:
                logfile.write('\tFailed to append all data for player (error): {0} ({1})\n'.format(key, value))
                logfile.write('\t{0}\n'.format(str(e)))

    df = pd.DataFrame()
    df['Name'] = names
    df['Id'] = ids
    df['height'] = heights
    df['weight'] = weights
    df['PG'] = pgs
    df['SG'] = sgs
    df['SF'] = sfs
    df['PF'] = pfs
    df['CE'] = ces
    df['salary'] = salaries
    df['shoots'] = shoots
    df['debut'] = debuts
    df['birth'] = births
    df['experience'] = experiences

    logfile.close()
    return df


def _getStaticPlayerData(player_id):
    """
    Gets the static data about a player by scraping
    it from https://www.basketball-reference.com.

    Parameters
    ----------
    The player ID to look for in basketball-reference.com

    Returns
    -------
    dictionary containing 'height', 'weight', 'first_pos', 'second_pos',
    'hand', 'birth', 'debut' statistics of the player
    """

    url = f"https://www.basketball-reference.com/players/{player_id[0:1].lower()}/{player_id}.html"
    try:
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        stats = [p.getText() for p in soup.find('div', {'id': 'meta'}).findAll('p')]
    except Exception:
        raise Exception('Player {0} not found on basketball-reference.com'.format(player_id))
    stats_dict = {
        'weight': None,
        'height': None,
        'hand': None,
        'birth': None,
        'debut': None,
        'experience': None,
        'Point Guard': -1,
        'Shooting Guard': -1,
        'Small Forward': -1,
        'Power Forward': -1,
        'Center': -1

    }
    for stat in stats:
        stat = re.sub('[^a-zA-Z0-9,]+', ' ', stat)
        try:
            if "Position" in stat and "Shoots" in stat:
                stats_dict['hand'] = stat.split('Shoots')[1].strip()
                posarr = stat.split('Shoots')[0].replace('Position', '').split('and')
                if posarr[0].strip().endswith(','):
                    posarr[0] = posarr[0].strip()[:-1]
                for idx, e in enumerate(posarr[0].split(',')):
                    stats_dict[e.strip()] = idx + 1
                if len(posarr) == 2:
                    stats_dict[posarr[1].strip()] = len(posarr[0].split(',')) + 1
            elif "Shoots" in stat:
                stats_dict['hand'] = stat.replace('Shoots', '').strip()
            elif "Debut" in stat:
                stats_dict['debut'] = stat.replace('NBA Debut', '').strip()
            elif "Experience" in stat:
                stats_dict['experience'] = stat.replace('Experience', '').strip()
            elif "Career Length" in stat:
                stats_dict['experience'] = stat.replace('Career Length', '').strip()
            elif "Born" in stat:
                stats_dict['birth'] = stat.split('in')[0].replace('Born', '').strip()
            elif 'cm' in stat and 'kg' in stat and 'lb' in stat:
                hw = stat.split('lb')[1]
                stats_dict['height'] = hw.split('cm')[0].strip()
                stats_dict['weight'] = hw.split('cm')[1].replace(',', '').replace('kg', '').strip()
            else:
                continue
        except Exception as e:
            print(e)
            raise Exception('Unable to parse line \'{0}\' for player {1}'.format(stat, player_id))
    return stats_dict


years = np.arange(2015, 2023)
for year in years:
    getAllStaticPlayerData(year).to_csv('static_player_stats_{0}.csv'.format(year))