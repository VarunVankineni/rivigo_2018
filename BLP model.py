# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:40:45 2018

@author: Varun
"""
import pandas as pd
import numpy as np
import scipy
from pulp import LpProblem,LpMinimize,LpVariable,LpStatus,LpBinary,LpSolverDefault,value
def waitCost(a):
    if(a>10):
        return (2*a)-10
    elif(a>=0):
        return a
    else:
        return 0

file = open('input005.txt', 'rb')
[t,s,ow] = [int(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
pitarray = np.zeros((t,3)).astype(str)
triparray = np.zeros((s,4)).astype(str)
for i in range(t):
    pitarray[i,:] = [str(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
for i in range(s):
    triparray[i,:] = [str(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
pit = pd.DataFrame(pitarray,columns = ['Pitstop_1','Pitstop_2','Travel_Time'])
trip = pd.DataFrame(triparray,columns = ['TripId','From_Pitstop','Departure_Time','To_Pitstop'])
trip['Departure_Time']= trip['Departure_Time'].astype(int)
del pitarray,triparray

trip.sort_values('Departure_Time',inplace = True)
trip.reset_index(inplace = True,drop = True)

axis0 = sorted(pit['Pitstop_1'].unique().tolist())
axis1 = sorted(pit['Pitstop_2'].unique().tolist())

dum = pd.DataFrame(np.zeros((len(axis0),len(axis1))).astype(int),index = axis0, columns = axis1)
for i in range(t):
    dum.loc[pit.iloc[i,0],pit.iloc[i,1]] = pit.iloc[i,2]
pit = dum.astype(int)
del dum,axis0,axis1

trip["Arrival_Time"] =  0
for i in range(s):
    trip.iloc[i,-1] = trip.iloc[i,2]+ pit.loc[trip.iloc[i,1],trip.iloc[i,3]]  
    
trip1 = trip[trip["From_Pitstop"]=='A']
trip2 = trip[trip["From_Pitstop"]=='B']
trip1.reset_index(inplace = True,drop = True)
trip2.reset_index(inplace = True,drop = True)
s1 = len(trip1)
s2 = len(trip2)
costAB = np.zeros((s1,s2+1)).astype(int) -1
costBA = np.zeros((s2,s1+1)).astype(int) -1

for i in range(s1):
    arrival = trip1.iloc[i,-1]
    dum = trip2[trip2["Departure_Time"]>=arrival]
    dum = dum["Departure_Time"]
    dum = dum - arrival
    if(dum.empty==False):
        for j in range(len(dum)):
            dum.iloc[j] = waitCost(dum.iloc[j])
        costAB[i,dum.index] = dum
    costAB[i,-1] = waitCost(ow-arrival)
for i in range(s2):
    arrival = trip2.iloc[i,-1]
    dum = trip1[trip1["Departure_Time"]>=arrival]
    dum = dum["Departure_Time"]
    dum = dum - arrival
    if(dum.empty==False):
        for j in range(len(dum)):
            dum.iloc[j] = waitCost(dum.iloc[j])
        costBA[i,dum.index] = dum
    costBA[i,-1] = waitCost(ow-arrival)
del arrival,dum

big = max(10**int(np.ceil(np.log10(costAB[0,-1]))+2),10**int(np.ceil(np.log10(costBA[0,-1]))+2))
costAB[costAB==-1] = big
costBA[costBA==-1] = big
del big

var1 = np.zeros(costAB.shape,object)
var2 = np.zeros(costBA.shape,object)
prob = LpProblem("Optimize Cost", LpMinimize)
for i in range(s1):
    for j in range(s2+1):   
        var1[i,j] = LpVariable("AB"+str(i)+' '+str(j),0,1,cat='binary')
for i in range(s2):
    for j in range(s1+1):   
        var2[i,j] = LpVariable("BA"+str(i)+' '+str(j),0,1,cat='binary')
sf = s1**2 + s2**2

prob += np.sum(np.multiply(costAB,var1)) + np.sum(np.multiply(costBA,var2))

for i in range(s1):
    prob += np.sum(var1[i,:])+np.sum(var2[:,i]) == 1
for i in range(s2):
    prob += np.sum(var2[i,:])+np.sum(var1[:,i]) == 1
    
LpSolverDefault.msg = 1
prob.solve()
cost = int(value(prob.objective))

trip1["Match"] = -2
trip2["Match"] = -2

for i in range(s1):
    for j in range(s2):   
        if(var1[i,j].varValue==1):
            trip1.iloc[i,-1] = trip2.iloc[j,0] 
            break
    if(var1[i,s2].varValue==1):
        trip1.iloc[i,-1] = '-1'
        
for i in range(s2):
    for j in range(s1):   
        if(var2[i,j].varValue==1):
            trip2.iloc[i,-1] = trip1.iloc[j,0]
            break
    if(var2[i,s1].varValue==1):
        trip2.iloc[i,-1] = '-1'
        
trip1 = trip1[trip1["Match"]!=-2]
trip2 = trip2[trip2["Match"]!=-2]

s1 = len(trip1)
s2 = len(trip2)
print(str(s1+s2))

for i in range(s1):
    if(trip1.iloc[i,-1]=='-1'):
        ender = str(-1)
    else:
        ender = trip1.iloc[i,3]
    print(trip1.iloc[i,1]+' '+trip1.iloc[i,3]+' '+trip1.iloc[i,0]+' '+trip1.iloc[i,-1]+' '+ender)
    
for i in range(s2):
    if(trip2.iloc[i,-1]=='-1'):
        ender = str(-1)
    else:
        ender = trip2.iloc[i,3]
    print(trip2.iloc[i,1]+' '+trip2.iloc[i,3]+' '+trip2.iloc[i,0]+' '+trip2.iloc[i,-1]+' '+ender)
    
print(cost)








