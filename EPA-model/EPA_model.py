"""
Scipt for calculation of Expected point added (EPA)

Dependent of NFLdb (python 2) don't run

Expected point added treat each combination of yards to go, down and yardline as different states and estimate the points added/subtracted by different states when game progress

The model below ulitize a logistic model similar to Yurko et al. (2018): https://arxiv.org/abs/1802.00998

"""


import csv
import math
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
import nfldb
from sklearn.linear_model import LogisticRegression

EPA_choice = np.zeros((99,4))
x = []
y = []

down_array = []
result_array = []
ytg_array = []
yardline_array = []

time_array = []
db = nfldb.connect()
q = nfldb.Query(db) 
games = q.game(season_year=[2016,2017] ).as_games() #Trained using 2016-17 data
next_drive_result = ''


for game in games:
    for drive in game.drives:
        if(next_drive_result == 'End of Half' or drive.result == 'End of Half' or next_drive_result == 'End of Game' or drive.result == 'End of Game'):
            continue
        
        for play in drive.plays: #fetch drive outcome from data
            
            first_down = 0
            second_down = 0
            third_down = 0
            fourth_down = 0    

            yard_str = str(play.yardline)
            yard_split_str = yard_str.split()
            pos_indicate = yard_split_str[0]

            if str(game.home_team) == str(play.pos_team):
                opp_team = str(game.away_team)
            else:
                opp_team = str(game.home_team)




            if pos_indicate == 'OWN':
                yardlinefromstr = int(yard_split_str[1])
            elif pos_indicate == 'OPP':
                yardlinefromstr = 100 - int(yard_split_str[1])
            else:
                yardlinefromstr = 50
            if(play.down <= 4 and play.down != None):

                

                if(drive.result == 'Touchdown'): 
                    y.append(6)
                    result_array.append(6)
                elif(drive.result == 'Field Goal'):
                    y.append(5)
                    result_array.append(5)
                elif(drive.result == 'Safety'):
                    y.append(2)    
                    result_array.append(2)
                else:
                    if(next_drive_result == 'Touchdown'):
                        y.append(0)
                        result_array.append(0)
                    elif(next_drive_result == 'Field Goal'):
                        y.append(1)
                        result_array.append(1)
                    elif(next_drive_result == 'Safety'):
                        y.append(4)    
                        result_array.append(4)
                    else:
                        y.append(3)
                        result_array.append(3)


                    
                    
                if(play.down == 1):
                    first_down = 1
                elif(play.down == 2):
                    second_down = 1
                elif(play.down == 3):
                    third_down = 1
                else:
                    fourth_down = 1

                x.append([yardlinefromstr,first_down, second_down, third_down, fourth_down ,play.yards_to_go])
                
                down_array.append(int(play.down))
                ytg_array.append(int(play.yards_to_go))
                yardline_array.append(int(yardlinefromstr))
                
                time_array.append(play.time.elapsed)
                
        next_drive_result = drive.result          

output_df = pd.DataFrame({'down':down_array,'ytg':ytg_array,'yardline':yardline_array, 'result':result_array, 'time': time_array})
output_df['result'] = output_df['result'].astype('category')
output_df['down'] = output_df['down'].astype('category')



"""
Perform multinomial logistic regression using sklearn
"""

y, X = dmatrices('result ~ time + down + np.log(ytg) + yardline + np.log(ytg):down + yardline:down', output_df, return_type = 'dataframe')
X_train, X_test, y_train, y_test, y_series_train, y_series_test = train_test_split(X, y, output_df['result'])

clf = LogisticRegression(random_state=0,multi_class='multinomial',max_iter=15000,solver='lbfgs')
clf.fit(X_train,y_series_train)


down_output = 1


"""
For each 1st to 4th down create a lookup table from model depend on yards to go and yardline
"""

output_array = []
for yardline_iter in range(1,100):
    output_row = []
    for ytg_iter in range(1,31):
        ytg_output = ytg_iter
        yardline_output = yardline_iter 
        predict_proba = clf.predict_proba(np.array([1,0,0,0,0,np.log(ytg_output),0,0,0,yardline_output,0,0,0 ]).reshape(1,-1))


        output_row.append(-7*predict_proba[0][0]-3*predict_proba[0][1]-2*predict_proba[0][2]+2*predict_proba[0][4]+3*predict_proba[0][5]+7*predict_proba[0][6]) 
    
    output_array.append(output_row)
np.savetxt('EPA_first.csv',output_array,delimiter=',')    

output_array = []
for yardline_iter in range(1,100):
    output_row = []
    for ytg_iter in range(1,31):
        ytg_output = ytg_iter
        yardline_output = yardline_iter 
        predict_proba = clf.predict_proba(np.array([1,1,0,0,0,np.log(ytg_output),np.log(ytg_output),0,0,yardline_output,yardline_output,0,0 ]).reshape(1,-1))


        output_row.append(-7*predict_proba[0][0]-3*predict_proba[0][1]-2*predict_proba[0][2]+2*predict_proba[0][4]+3*predict_proba[0][5]+7*predict_proba[0][6])
    
    output_array.append(output_row)
np.savetxt('EPA_second.csv',output_array,delimiter=',') 

output_array = []
for yardline_iter in range(1,100):
    output_row = []
    for ytg_iter in range(1,31):
        ytg_output = ytg_iter
        yardline_output = yardline_iter 
        predict_proba = clf.predict_proba(np.array([1,0,1,0,0,np.log(ytg_output),0,np.log(ytg_output),0,yardline_output,0,yardline_output,0 ]).reshape(1,-1))


        output_row.append(-7*predict_proba[0][0]-3*predict_proba[0][1]-2*predict_proba[0][2]+2*predict_proba[0][4]+3*predict_proba[0][5]+7*predict_proba[0][6])
    
    output_array.append(output_row)
np.savetxt('EPA_third.csv',output_array,delimiter=',') 

output_array = []
for yardline_iter in range(1,100):
    output_row = []
    for ytg_iter in range(1,31):
        ytg_output = ytg_iter
        yardline_output = yardline_iter 
        predict_proba = clf.predict_proba(np.array([1,0,0,1,0,np.log(ytg_output),0,0,np.log(ytg_output),yardline_output,0,0,yardline_output]).reshape(1,-1))


        output_row.append(-7*predict_proba[0][0]-3*predict_proba[0][1]-2*predict_proba[0][2]+2*predict_proba[0][4]+3*predict_proba[0][5]+7*predict_proba[0][6])
    
    output_array.append(output_row)
np.savetxt('EPA_fourth.csv',output_array,delimiter=',') 
