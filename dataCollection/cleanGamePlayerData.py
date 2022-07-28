import re
import numpy as np

# Samples to use for testing (#2 should not pass, does not have 0 or 1 for starter/home
'''
29402,Sandro Mamukelashvili,22:47,3,6,.500,1,3,.333,1,2,.500,3,1,4,2,0,2,1,1,8,-20,0,1,mamuksa01,202204010MIL,.581,.583,.500,.333,15.4,6.4,11.4,9.6,0.0,8.6,12.7,14.2,126,151,-1.8

29402,Sandro Mamukelashvili,,,,,,,,,,,,,,,,,,,,,,,mamuksa01,202204010MIL,,,,,,,,,,,,,,,

0,Paul Millsap,36:00,7,15,.467,2,6,.333,3,4,.750,1,7,8,3,0,0,2,4,19,-22,1,1,millspa01,201510270ATL,.567,.533,.400,.267,3.1,16.7,10.8,14.5,0.0,0.0,10.7,24.1,107,113,-0.5

7,Lamar Patterson,18:18,1,1,1.000,1,1,1.000,2,2,1.000,0,1,1,2,0,0,0,4,5,+10,0,1,pattela01,201510270ATL,1.330,1.500,1.000,2.000,0.0,4.7,2.6,15.3,0.0,0.0,0.0,4.8,258,121,2.1

42,J.R. Smith,30:27,3,10,.300,0,2,.000,2,4,.500,1,4,5,3,1,0,0,4,8,0,1,0,smithjr01,201510270CHI,.340,.300,.200,.400,3.1,13.7,8.1,14.2,1.6,0.0,0.0,16.6,87,99,-5.0

1511,Seth Curry,3:16,2,2,1.000,0,0,,0,0,,0,0,0,1,0,0,0,0,4,+5,0,1,curryse01,201511030SAC,1.000,1.000,.000,.000,0.0,0.0,0.0,-1000.0,0.0,0.0,0.0,28.1,203,117,34.6

5597,Chris Andersen,4:12,1,1,1.000,1,1,1.000,0,0,,0,1,1,1,0,0,0,0,3,+5,0,0,anderch01,201511250DET,1.167,1.167,.333,.000,0.0,22.4,11.8,-500.0,13.2,0.0,0.0,34.6,232,86,76.5

8394,Matt Bonner,0:15,0,0,,0,0,,0,0,,0,0,0,0,1,0,0,0,0,+3,0,0,bonnema01,201512090TOR,,,,,0.0,0.0,0.0,0.0,100.0,0.0,,0.0,0,-345,247.8

17790,Dwight Howard,18:39,0,1,.000,0,0,,9,14,.643,4,7,11,1,1,0,2,4,9,-6,0,0,howardw01,202103270LAC,.628,.000,.000,14.000,27.8,46.2,37.3,6.6,2.6,0.0,21.8,21.9,125,116,0.6

'''

def cleanGamePlayerDataFile(filename, delete_all=False):

    line_regex = \
        "^[\d]{1,5},.*,([\d]{1,2}:[0-6][\d])*,([\d]*,){2}((1\.000)|(\.[\d]{3}))?,([\d]*,){2}((1\.000)|(\.[\d]{3}))?,([\d]*,){2}((1\.000)|(\.[\d]{3}))?,([\d]*,){9}(((\+|-)[\d]+)|0)?,([0-1],){2}[a-z]{1,10}[\d]{2},[\d]{8}0[A-Z]{3},((\d?\.[\d]{3})?,){3}(\d*\.[\d]{3})?,(-?[\d\.]*,){8}(-?[\d]*,){2}(-?[\d\.]*)$"

    expected_header = \
        ",Name,MP,FG,FGA,FG%,3P,3PA,3P%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,+/-,started,home,playerid,gameid,TS%,eFG%,3PAr,FTr,ORB%,DRB%,TRB%,AST%,STL%,BLK%,TOV%,USG%,ORtg,DRtg,BPM"

    fn_split = filename.rsplit('.', 1)
    if len(fn_split) != 2 or fn_split[1] != 'csv':
        raise Exception('Invalid filename given ({0})'.format(filename))

    data = open(filename, 'r', encoding='utf-8')
    clean_data = open(fn_split[0] + '_clean.' + fn_split[1], 'w', encoding='utf-8')
    data_lines = data.readlines()

    # Pops and writes header line to clean file
    header = data_lines.pop(0)
    if header.strip() != expected_header:
        raise Exception('header of file does not match expected header')
    clean_data.write(header.strip() + '\n')
    print('Line {0} is clean'.format('header'))

    for line in data_lines:
        if not re.match(line_regex, line.strip()):
            print(u'Line {0} is not clean:\n\t{1}'.format(line.split(',')[0], line.strip()))
            if delete_all:
                continue
            else:
                usrin = input('What would you like to do with this line ([d]elete/[e]dit/[k]eep)? ')
                if usrin.lower() == 'd':
                    continue
                elif usrin.lower() == 'e':
                    edited = input('Please input the correct line: ')
                    clean_data.write(edited.strip() + '\n')
                elif usrin.lower() == 'k':
                    clean_data.write(line.strip() + '\n')
                else:
                    clean_data.write(line.strip() + '\n')
        else:
            print('Line {0} is clean'.format(line.split(',')[0]))
            clean_data.write(line.strip() + '\n')

    clean_data.close()

years = np.arange(2015, 2022)
for year in years:
    cleanGamePlayerDataFile('../data/gamePlayerData/game_data_player_stats_{0}.csv'.format(year))

# cleanGamePlayerDataFile('../data/gamePlayerData/game_data_player_stats_{0}.csv'.format(2015))