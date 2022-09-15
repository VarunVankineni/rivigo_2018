# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 23:45:08 2018

@author: Varun
"""
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment as assign

def timer(a):
    if(a>10):
        return (2*a)-10
    elif(a>=0):
        return a

    
file = open('E:/Acads/9th sem/rivigo/input001.txt', 'rb')

[t,s,ow] = [int(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
pitstop = np.zeros((t,3)).astype(str)
schedule = np.zeros((s,4)).astype(str)
for i in range(t):
    pitstop[i,:] = [str(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
for i in range(s):
    schedule[i,:] = [str(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
pitdf = pd.DataFrame(pitstop,columns = ['Pitstop_1','Pitstop_2','Travel_Time'])
schdf = pd.DataFrame(schedule,columns = ['TripId','From_Pitstop','Departure_Time','To_Pitstop'])
schdf['Departure_Time']= schdf['Departure_Time'].astype(int)

del pitstop,schedule

schdf.sort_values('Departure_Time',inplace = True)
schdf.reset_index(inplace = True,drop = True)

axis0 = sorted(pitdf['Pitstop_1'].unique().tolist())
axis1 = sorted(pitdf['Pitstop_2'].unique().tolist())

dum = pd.DataFrame(np.zeros((len(axis0),len(axis1))).astype(int),index = axis0, columns = axis1)
for i in range(t):
    dum.loc[pitdf.iloc[i,0],pitdf.iloc[i,1]] = pitdf.iloc[i,2]
pitdf = dum.astype(int)
del dum

schdf["Arrival_Time"] =  0

for i in range(s):
    schdf.iloc[i,-1] = schdf.iloc[i,2]+ pitdf.loc[schdf.iloc[i,1],schdf.iloc[i,3]]

costmat = np.zeros((s,s)).astype(int)
costmat = costmat-1

for i in range(s):
    arrival = schdf.iloc[i,-1]
    arrstop = schdf.iloc[i,3]
    dum = schdf[schdf["Departure_Time"]>=arrival]
    dum = dum[dum["From_Pitstop"]==arrstop]
    dum = dum["Departure_Time"]
    dum = dum - arrival
    if(dum.empty==False):
        for j in range(len(dum)):
            dum.iloc[j] = timer(dum.iloc[j])
        costmat[i,dum.index] = dum
        
costmat[costmat==-1]=10**int(2*np.ceil(np.log10(costmat[0,-1])))
matchmat = assign(costmat)




