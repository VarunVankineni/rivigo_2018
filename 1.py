# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:41:07 2018

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
    
    
file = open('input2b.txt', 'rb')
wfile = open("output2b.txt","w")

N = int(file.readline())
wfile.write(str(N)+'\n')
for cases in range(N):
    currentcase,lines = file.readline().decode("utf-8").rstrip().split(" ")
    pitlist = file.readline().decode("utf-8").rstrip().split(" ")
    
    [t,s,ow,r] = [int(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
    
    pitarray = np.zeros((t,3)).astype(str)
    triparray = np.zeros((s,6)).astype(str)
    
    for i in range(t):
        pitarray[i,:] = [str(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
    for i in range(s):
        triparray[i,:] = [str(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
    pit = pd.DataFrame(pitarray,columns = ['Pitstop_1','Pitstop_2','Travel_Time'])
    trip = pd.DataFrame(triparray,columns = ['TripId','Next_Pitstop','Starts_After','Destiny','Destination','Arrive_By'])
    trip['Starts_After']= trip['Starts_After'].astype(int)
    trip['Arrive_By']= trip['Arrive_By'].astype(int)
    del pitarray,triparray
    
    trip.sort_values('Arrive_By',inplace = True)
    trip.reset_index(inplace = True,drop = True)
        
    axis0 = sorted(pit['Pitstop_1'].unique().tolist())
    axis1 = sorted(pit['Pitstop_2'].unique().tolist())
    
    dum = pd.DataFrame(np.zeros((len(axis0),len(axis1))).astype(int),index = axis0, columns = axis1)
    for i in range(t):
        dum.loc[pit.iloc[i,0],pit.iloc[i,1]] = pit.iloc[i,2]
    pit = dum.astype(int)
    del dum,axis0,axis1
    
    
    trip["Stops"] = trip["Destination"].apply(ord) - trip["Next_Pitstop"].apply(ord)
    trip['Destiny'] = trip["Destination"]
    runs = np.sum(trip["Stops"].apply(abs))
    i=0
    k = s
    trip = trip.append(pd.DataFrame(np.zeros((runs-s,7)), columns = trip.columns),ignore_index = True)
    for i in range(runs):
        stops = int(trip.iat[i,-1])
        sign = np.sign(stops)
        if(abs(stops)!=1):
            newrow = trip.iloc[i,:]
            newrow[1] = chr(sign+ ord(newrow[1]))
            trip.iat[i,4] = newrow[1]
            newrow[2] = newrow[2] + pit.loc[trip.iat[i,1],trip.iat[i,4]]
            trip.iat[i,-1] = sign 
            newrow[-1] = stops - sign
            trip.iloc[k,:] = newrow
            k+=1

    
    snew = len(trip)
    for i in range(snew):
        trip.iloc[i,-2] = trip.iloc[i,2]+ pit.loc[trip.iloc[i,1],trip.iloc[i,4]]  
    trip.drop('Stops',axis = 1, inplace = True)
    
    lpps = len(pitlist)-1
    matchlist = [0 for x in range(lpps)]
    cost = 0
    for j in range(lpps):
        trip1 = trip[trip["Next_Pitstop"]==pitlist[j]]
        trip1 = trip1[trip1["Destination"]==pitlist[j+1]]
        trip2 = trip[trip["Next_Pitstop"]==pitlist[j+1]]
        trip2 =trip2[trip2["Destination"]==pitlist[j]]
        trip1.reset_index(inplace = True,drop = True)
        trip2.reset_index(inplace = True,drop = True)
        s1 = len(trip1)
        s2 = len(trip2)
        costAB = np.zeros((s1,s2+1)).astype(int) -1
        costBA = np.zeros((s2,s1+1)).astype(int) -1
        for i in range(s1):
            arrival = trip1.iloc[i,-1]
            dum = trip2[trip2["Starts_After"]>=arrival]
            dum = dum["Starts_After"]
            dum = dum - arrival
            if(dum.empty==False):
                for k in range(len(dum)):
                    dum.iloc[k] = waitCost(dum.iloc[k])
                costAB[i,dum.index] = dum
            costAB[i,-1] = waitCost(ow-arrival)
        for i in range(s2):
            arrival = trip2.iloc[i,-1]
            dum = trip1[trip1["Starts_After"]>=arrival]
            dum = dum["Starts_After"]
            dum = dum - arrival
            if(dum.empty==False):
                for k in range(len(dum)):
                    dum.iloc[k] = waitCost(dum.iloc[k])
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
            for k in range(s2+1):   
                var1[i,k] = LpVariable("AB"+str(i)+' '+str(k),0,1,cat='binary')
        for i in range(s2):
            for k in range(s1+1):   
                var2[i,k] = LpVariable("BA"+str(i)+' '+str(k),0,1,cat='binary')
        sf = s1**2 + s2**2
        
        prob += np.sum(np.multiply(costAB,var1)) + np.sum(np.multiply(costBA,var2))
        
        for i in range(s1):
            prob += np.sum(var1[i,:])+np.sum(var2[:,i]) == 1
        for i in range(s2):
            prob += np.sum(var2[i,:])+np.sum(var1[:,i]) == 1
            
        LpSolverDefault.msg = 1
        prob.solve()
        cost +=int(value(prob.objective))
        
        trip1["Match"] = -2
        trip2["Match"] = -2
        
        for i in range(s1):
            for k in range(s2):   
                if(var1[i,k].varValue==1):
                    trip1.iloc[i,-1] = trip2.iloc[k,0] 
                    break
            if(var1[i,s2].varValue==1):
                trip1.iloc[i,-1] = '-1'
                
        for i in range(s2):
            for k in range(s1):   
                if(var2[i,k].varValue==1):
                    trip2.iloc[i,-1] = trip1.iloc[k,0]
                    break
            if(var2[i,s1].varValue==1):
                trip2.iloc[i,-1] = '-1'
        
        matchlist[j] = trip1.append(trip2, ignore_index = True).sort_values('Arrive_By')
        
    
        
    trip = matchlist[0]
    for i in range(1,lpps):
        trip = trip.append(matchlist[i]) 
    trip.sort_values(['TripId','Starts_After'], inplace = True)
    
    matches = len(trip[trip["Match"]!=-2])
    
    
    wfile.write(currentcase + ' ' + str(snew+matches)+'\n')
    wfile.write(str(snew)+' '+str(matches)+'\n')
    for i in range(snew):
         wfile.write(trip.iloc[i,0]+' '+trip.iloc[i,1]+' '+str(int(trip.iloc[i,2]))+'\n')
    
    trip = trip[trip["Match"]!=-2]
    
    for i in range(matches):
        if(trip.iloc[i,-1]=='-1'):
            ender = str(-1)
        else:
            ender = trip.iloc[i,4]            
        wfile.write(trip.iloc[i,1]+' '+trip.iloc[i,4]+' '+trip.iloc[i,0]+' '+trip.iloc[i,-1]+' '+ender+'\n')
    
    wfile.write(str(cost)+'\n')

wfile.close()
file.close()






