import pandas as pd
import numpy as np
import bs4 as bs
from urllib.request import urlopen
from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster, Player
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

def getPlayerSalaryData(year):
    """
    Gets the players' salaries for the given year.

    Parameters
    ----------
    year : the year to scrape

    Returns
    -------
    a dataframe containing the player salaries and player ids
    """

    print('Scraping year {0}'.format(year))

    inputDF = pd.read_csv('../data/staticPlayerData/static_player_stats_{0}.csv'.format(year), index_col=0)
    playerids = inputDF['Id'].to_list()

    salaries = []
    for playerid in playerids:
        try:
            sal = scrapePlayerPastYearSalaryData(year, playerid)
            print('Successful: {0}, {1}'.format(playerid, sal))
            salaries.append(sal)
        except Exception as e:
            print('Failed: {0}, {1}'.format(playerid, np.nan))
            print(e)
            salaries.append(np.nan)

    df = inputDF
    # df['Id'] = inputDF['Id']
    df['salary'] = salaries
    print('Done')
    df.to_csv('../data/staticPlayerData/static_player_stats_{0}.csv'.format(year))
    return df

def scrapePlayerPastYearSalaryData(year, playerid):
    """
    Scrapes a given player's salary for a given year in the past.

    Parameters
    ----------
    year : the year to scrape
    playerid : the player id to scrape

    Returns
    -------
    a numerical salary value
    """

    regex = r"<tr ><th [^<>]* data-stat=\"season\" >" + str(year-1) + "-" + str(year)[2:4] \
            + r"<\/th>[^\n]*<td [^<>]* data-stat=\"salary\" [^<>]*>\$([\d,]+)</td></tr>\n"
    regex = r"<tr ><th [^<>]* data-stat=\"season\" >" + str(year-1) + "-" + str(year)[2:4] \
            + r"<\/th>[^\n]*<td [^<>]* data-stat=\"salary\" [^<>]*>(\$([\d,]+)|(<? \$Minimum))</td></tr>\n"
    MIN_NBA_SAL = 1e5

    url = f"https://www.basketball-reference.com/players/{playerid[0:1].lower()}/{playerid}.html"
    try:
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        matches = re.findall(regex, str(soup.find('div', {'id': 'all_all_salaries'})))
        if len(matches) > 1:
            #print('len > 1 for {0}'.format(playerid))
            max_sal = -1
            for m in matches:
                if isinstance(m, tuple) and not len(m) == 0:
                    m = m[0]
                if not m:
                    continue
                elif "Minimum" in m:
                    new_sal = MIN_NBA_SAL
                else:
                    new_sal = int(m.replace(',', '').replace('$', ''))
                max_sal = new_sal if new_sal > max_sal else max_sal
            return max_sal
        elif len(matches) == 0:
            #print('len == 0 for {0}'.format(playerid))
            raise Exception()
        else:
            if isinstance(matches[0], tuple) and not len(matches[0]) == 0:
                matches[0] = matches[0][0]

            if not matches[0]:
                raise Exception()
            elif "Minimum" in matches[0]:
                new_sal = MIN_NBA_SAL
            else:
                new_sal = int(matches[0].replace(',', '').replace('$', ''))

            return new_sal

    except Exception:
        raise Exception('Player {0} salary not found on basketball-reference.com for year {1}'.format(playerid, year))


def scrapePlayerSalaryData(playerid, team_abbr):
    """
        Scrapes a given player's salary for the current year

        Parameters
        ----------
        playerid : the player id to scrape
        team_abbr: the team that the player is playing on

        Returns
        -------
        a numerical salary value
    """

    url = f"https://www.basketball-reference.com/players/{playerid[0:1].lower()}/{playerid}.html"
    year = 2023;
    try:
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        # regex = r"<tr ><th [^<>]* data-stat=\"season\" >" + str(year - 1) + "-" + str(year)[2:4] \
        #         + r"<\/th>[^\n]*<td [^<>]* data-stat=\"salary\" [^<>]*>(\$([\d,]+)|(<? \$Minimum))</td></tr>\n"
        regex = r"<table [^<>]* id=\"contracts_" + team_abbr.lower() + r"\" [^<>]*>[\W\w]*<tr>[\W\w]*<td[^<>]*>" \
                + r"<span [^<>]*>(\$([\d,]+)|(<? \$Minimum))</span></td>\n*</tr>\n*</table>"
        matches = re.findall(regex, str(soup.find('div', {'id': 'all_contract'})))
        if len(matches) > 1:
            # print('len > 1 for {0}'.format(playerid))
            max_sal = -1
            for m in matches:
                if isinstance(m, tuple) and not len(m) == 0:
                    m = m[0]
                if not m:
                    continue
                elif "Minimum" in m:
                    new_sal = 1e5
                else:
                    new_sal = int(m.replace(',', '').replace('$', ''))
                max_sal = new_sal if new_sal > max_sal else max_sal
            return max_sal
        elif len(matches) == 0:
            # print('len == 0 for {0}'.format(playerid))
            raise Exception()
        else:
            if isinstance(matches[0], tuple) and not len(matches[0]) == 0:
                matches[0] = matches[0][0]

            if not matches[0]:
                raise Exception()
            elif "Minimum" in matches[0]:
                new_sal = 1e5
            else:
                new_sal = int(matches[0].replace(',', '').replace('$', ''))

            return new_sal

    except Exception:
        raise Exception('Player {0} salary not found on basketball-reference.com for year {1}'.format(playerid, year))




# years = np.arange(2015, 2023)
# for year in years:
#     getAllStaticPlayerData(year).to_csv('static_player_stats_{0}.csv'.format(year))
# print(scrapePlayerPastYearSalaryData(2022, 'duranke01'))
#print(scrapePlayerSalaryData('rosste01', 'ORL'))