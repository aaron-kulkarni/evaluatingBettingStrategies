import bs4
from urllib.request import urlopen
import re
import pandas as pd
from sportsipy.nba.boxscore import Boxscores
import datetime as dt


def getGameAttendanceReferees(str_time, end_time):
    """
    Creates a dataframe with the game id, attendance and referees
    for each game between the start and end date.

    Parameters
    ----------
    str_time : the start time in YYYYMMDD format
    end_time : the end time in YYYYMMDD format

    Returns
    -------
    A pandas dataframe
    """

    if not str_time or not end_time:
        raise Exception('Invalid start or end times')

    rows = [scrapeGameAttendanceReferees(g['boxscore']) for k, v in
            Boxscores(dt.datetime.strptime(str_time, '%Y%m%d'),
                      dt.datetime.strptime(end_time, '%Y%m%d')).games.items()
            for g in v]

    return pd.DataFrame(rows)


def scrapeGameAttendanceReferees(gameid=None):
    """
    Scrapes the referees and number of people attending the game from
    basketball-reference.com for a particular game id

    Parameters
    ----------
    gameid : the basketball-reference.com boxscore id of the game to be scraped

    Returns
    -------
    A dictionary as such: {
        'gameid' : 123456789ABC,
        'ref' : ['referee 1 name', referee 2 name', ...],
        'att' : 12345
    }
    """

    if gameid is None or not re.match(r"^[\d]{9}[A-Z]{3}$", gameid):
        raise Exception("Issue with game ID")

    url = 'https://www.basketball-reference.com/boxscores/{0}.html'.format(gameid)
    outdict = {'gameid': gameid, 'ref': [], 'att': 0}

    try:
        soup = bs4.BeautifulSoup(urlopen(url), features='lxml')
        divs = [div.getText() for div in
                soup.find('div', {'id': 'content'}).findAll('div', recursive=False)[-2].findAll('div')]
        for div in divs:
            if 'Officials' in div:
                outdict['ref'] = div.replace('Officials:', '').replace('\xa0', '').strip().split(', ')
            elif 'Attendance' in div:
                outdict['att'] = int(div.replace('Attendance:', '').replace('\xa0', '').strip().replace(',', ''))
        print("Successfully added game: {0}".format(gameid))
    except Exception as e:
        print("Failed to add game: {0}".format(gameid))
    # print(outdict)
    return outdict


# Tests scrapeGameAttendanceReferees
# print(scrapeGameAttendanceReferees('202110190MIL'))

yeardates = {
    2015: ('20141028', '20150615'),
    2016: ('20151027', '20160619'),
    2017: ('20161025', '20170612'),
    2018: ('20171017', '20180608'),
    2019: ('20181016', '20190613'),
    2020: ('20191022', '20201011'),
    2021: ('20201222', '20210720'),
    2022: ('20211019', '20220616')
}

# Check format of output with the following regex:
# [\d]{1,4},20[1-2][\d]{3}[0-3][\d]0[A-Z]{3},((\"\[('[A-Za-z \.-]+',? ?)+\]\")|(\['[A-Za-z \.-]+'\])),[\d]+
for year, dates in yeardates.items():
    getGameAttendanceReferees(dates[0], dates[1]).to_csv('game_attendance_ref_{0}.csv'.format(year))
    print('Created CSV for year {0} ({1} - {2})'.format(year, dates[0], dates[1]))
