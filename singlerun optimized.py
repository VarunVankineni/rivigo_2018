    # -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:41:07 2018

@author: Varun
"""
import pandas as pd
import numpy as np
import scipy
import os
from multiprocessing import Pool,Process,Queue,freeze_support
import time
from pulp import LpProblem,LpMinimize,LpVariable,LpStatus,LpBinary,LpSolverDefault,value

def waitCost(a):
    if(a>10):
        return (2*a)-10
    elif(a>=0):
        return a
    else:
        return 0
def quickJourney(x):
    time = 0
    a = x[0]
    b = x[1]
    sign = np.sign(ord(b)-ord(a))
    while(a!=b):
        nextstop = chr(ord(a)+sign) 
        time+=pit.at[a,nextstop]
        a = nextstop
    return time

def jTimes(j):
    axis0 = sorted(j['Pitstop_1'].unique().tolist())
    axis1 = sorted(j['Pitstop_2'].unique().tolist())   
    dum = pd.DataFrame(np.zeros((len(axis0),len(axis1)),int)-1,index = axis0, columns = axis1)
    for i,rows in j.iterrows():
        dum.at[rows[0],rows[1]] = rows[2]
    dum = dum.astype(int)            
    return dum
    
def sortReset(df,col):
    df.sort_values(col, inplace = True)
    df.reset_index(inplace = True,drop = True)
    return trip

def randomBufAllocate(trip,buffers,usedbuffers):
    for i, rows in trip.iterrows():
        tripid = rows[0]
        avabuf = buffers.at[tripid]
        usedbuf = np.random.randint(0,avabuf+1)
        usedbuffers[tripid] += usedbuf
        trip.iat[i,-2] = usedbuffers[tripid]
        buffers.at[tripid] = avabuf - usedbuf 
    return trip
def neighbourBufAllocate(trip,buffers,usedbuffers,curindibuffers,base):
    for i, rows in trip.iterrows():
        tripid = rows[0]
        avabuf = buffers.at[tripid]
        lastbuf = curindibuffers[i]
        limits = 1+int(avabuf/base)
        lower = max(0,lastbuf-limits)
        upper = min(lastbuf+limits,avabuf)
        if(lower>=upper):
            usedbuf = np.random.randint(0,avabuf+1)
        else:
            usedbuf = np.random.randint(lower,upper+1)
        curindibuffers[i] = usedbuf
        usedbuffers[tripid] += usedbuf
        trip.iat[i,-2] = usedbuffers[tripid]
        buffers.at[tripid] = avabuf - usedbuf
    return trip,curindibuffers

def curBufferCost(trip,lpps,version = 0):
    curcost = 0
    matchlist = [0 for x in range(lpps)]
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
            arrival = trip1.iat[i,-1]
            dum = trip2[trip2["Starts_After"]>=arrival]
            dum = dum["Starts_After"]
            dum = dum - arrival
            if(dum.empty==False):
                for k in range(len(dum)):
                    dum.iat[k] = waitCost(dum.iat[k])
                costAB[i,dum.index] = dum
            costAB[i,-1] = waitCost(ow-arrival)
        for i in range(s2):
            arrival = trip2.iat[i,-1]
            dum = trip1[trip1["Starts_After"]>=arrival]
            dum = dum["Starts_After"]
            dum = dum - arrival
            if(dum.empty==False):
                for k in range(len(dum)):
                    dum.iat[k] = waitCost(dum.iat[k])
                costBA[i,dum.index] = dum
            costBA[i,-1] = waitCost(ow-arrival)
        del arrival,dum
        
        big = max(10**int(np.ceil(np.log10(max(1,costAB[0,-1])))+2),10**int(np.ceil(np.log10(max(1,costAB[0,-1])))+2))
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
        
        prob += np.sum(np.multiply(costAB,var1)) + np.sum(np.multiply(costBA,var2))
        
        for i in range(s1):
            prob += np.sum(var1[i,:])+np.sum(var2[:,i]) == 1
        for i in range(s2):
            prob += np.sum(var2[i,:])+np.sum(var1[:,i]) == 1
    
        LpSolverDefault.msg = 1
        prob.solve()
        curcost +=int(value(prob.objective))
        if(version ==1):
            trip1["Match"] = -2
            trip2["Match"] = -2
            for i in range(s1):
                for k in range(s2):   
                    if(var1[i,k].varValue==1):
                        trip1.iat[i,-1] = trip2.iat[k,0] 
                        break
                if(var1[i,s2].varValue==1):
                    trip1.iat[i,-1] = '-1'
                    
            for i in range(s2):
                for k in range(s1):   
                    if(var2[i,k].varValue==1):
                        trip2.iat[i,-1] = trip1.iat[k,0]
                        break
                if(var2[i,s1].varValue==1):
                    trip2.iat[i,-1] = '-1'
            matchlist[j] = trip1.append(trip2, ignore_index = True).sort_values('Arrive_By')
    
    if(version==0):
        return curcost
    elif(version==1):    
        return curcost,matchlist
    
    
def findCostlist(mastertrip,masterbuffers,usedbuffers):
    temp_time = time.time()
    costlist = []
    while((time.time()-temp_time)<int(0.1*r)):
        trip = mastertrip.copy()
        buffers = masterbuffers.copy()
        usedbuffers[:] = 0
        trip = randomBufAllocate(trip,buffers,usedbuffers)
        currentbuf = trip["Buffer"]
        trip["Starts_After"] = trip["Starts_After"] + trip["Buffer"]    
        trip.drop('Buffer',axis =1, inplace = True)
        trip = trip[trip["Starts_After"]<ow]
        for i in range(len(trip)):
            trip.iat[i,-2] = trip.iat[i,2]+ pit.at[trip.iat[i,1],trip.iat[i,4]]  
        trip.drop('Stops',axis = 1, inplace = True)
        
        lpps = len(pitlist)-1
        cost = curBufferCost(trip,lpps,0)            
        costlist.append(cost)   
    return costlist,currentbuf

def solveForOptBuffer(mastertrip,masterbuffers,usedbuffers,indibuffers,Tmax,maxtime):
    mincost = 100000000000
    minglobal = mincost
    temp_time = time.time() + int(0.7*r)
    maxtime += int(0.9*r)
    while((time.time()<temp_time) and (time.time()<maxtime) ):
        base = 7+int(10*(time.time()-temp_time)/(r))
        T = Tmax / 2**(base-1)
        base+=1
        trip = mastertrip.copy()
        buffers = masterbuffers.copy()
        usedbuffers[:] = 0
        curindibuffers = indibuffers
        trip,curindibuffers = neighbourBufAllocate(trip,buffers,usedbuffers,curindibuffers,base)        
    
        currentbuf = trip["Buffer"]
        trip["Starts_After"] = trip["Starts_After"] + trip["Buffer"]    
        trip.drop('Buffer',axis =1, inplace = True)
        trip = trip[trip["Starts_After"]<ow]
        for i in range(len(trip)):
            trip.iat[i,-2] = trip.iat[i,2]+ pit.at[trip.iat[i,1],trip.iat[i,4]]  
        trip.drop('Stops',axis = 1, inplace = True)
        lpps = len(pitlist)-1
        cost = curBufferCost(trip,lpps,0)
        if(minglobal > cost):
            globalminbuf = currentbuf
            minglobal = cost
        
        diff = cost - mincost
        if(diff<0):
            mincost = cost
            indibuffers = curindibuffers.copy()
        else:
            if(np.random.uniform()<np.exp(-diff/T)):
                mincost = cost
                indibuffers = curindibuffers.copy()

    #output.put([globalminbuf,minglobal])
    return globalminbuf,minglobal

"""
Main Func
"""

file = open('E:/Acads/9th sem/rivigo/2b,c/input.txt', 'rb')
N = int(file.readline())
print(str(N)+'\n')
for cases in range(N):
    currentcase,lines = file.readline().decode("utf-8").rstrip().split(" ")
    pitlist = file.readline().decode("utf-8").rstrip().split(" ")
    
    [t,s,ow,r] = [int(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
    
    start_time = time.time()
    
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
    
    trip = sortReset(trip,'Arrive_By')
    pit = jTimes(pit)
    
    trip["Buffer"] = trip["Arrive_By"] - ( trip["Starts_After"] + trip.loc[:,['Next_Pitstop','Destination']].apply(quickJourney, axis = 1))
    buffers = trip.loc[:,["TripId","Buffer"]]
    buffers = buffers.set_index('TripId').sort_index()
    buffers = buffers.iloc[:,0]
    
    trip["Stops"] = trip["Destination"].apply(ord) - trip["Next_Pitstop"].apply(ord)
    trip['Destiny'] = trip["Destination"]
    runs = np.sum(trip["Stops"].apply(abs))
    
    
    trip = trip.append(pd.DataFrame(np.zeros((runs-s,8)), columns = trip.columns),ignore_index = True)
    k = s
    for i in range(runs):
        stops = int(trip.iat[i,-1])
        sign = np.sign(stops)
        if(abs(stops)!=1):
            newrow = trip.iloc[i,:]
            newrow[1] = chr(sign+ ord(newrow[1]))
            trip.iat[i,4] = newrow[1]
            newrow[2] = newrow[2] + pit.at[trip.iat[i,1],trip.iat[i,4]]
            trip.iat[i,-1] = sign 
            newrow[-1] = stops - sign
            trip.iloc[k,:] = newrow
            k+=1
    
    trip = sortReset(trip,['TripId','Starts_After'])
    
    mastertrip = trip.copy()
    masterbuffers = buffers.copy()
    usedbuffers = masterbuffers.copy()
    usedbuffers[:] = 0
    
    indibuffers = trip["Buffer"]
    indibuffers[:] = 0
    
    costlist,currentbuf = findCostlist(mastertrip.copy(),masterbuffers.copy(),usedbuffers)
    
    Tmean = np.mean(costlist)
    Tmax = 100*np.max([np.abs(i-Tmean) for i in costlist])
   
    globalminbuf, minglobal = solveForOptBuffer(mastertrip.copy(),masterbuffers.copy(),usedbuffers,indibuffers,Tmax,start_time)
            
    trip = mastertrip.copy()
    buffers = masterbuffers.copy()
    usedbuffers[:] = 0
    trip["Starts_After"] = trip["Starts_After"] + globalminbuf
    trip.drop('Buffer',axis =1, inplace = True)
    trip = trip[trip["Starts_After"]<ow]
    for i in range(len(trip)):
        trip.iat[i,-2] = trip.iat[i,2]+ pit.at[trip.iat[i,1],trip.iat[i,4]]  
    trip.drop('Stops',axis = 1, inplace = True)
    
    lpps = len(pitlist)-1
    cost, matchlist = curBufferCost(trip,lpps,version = 1)
    trip = matchlist[0]
    for i in range(1,lpps):
        trip = trip.append(matchlist[i]) 
    trip.sort_values(['TripId','Starts_After'], inplace = True)
    
    mastertripfinal = trip.copy()
    snew = len(trip)
    matches = len(trip[trip["Match"]!=-2])
    print(currentcase + ' ' + str(snew+matches+2)+'\n')
    print(str(snew)+' '+str(matches)+'\n')
    for i in range(snew):
         print(trip.iat[i,0]+' '+trip.iat[i,1]+' '+str(int(trip.iat[i,2]))+'\n')
    trip = trip[trip["Match"]!=-2]
    trip = trip.astype(str)
    for i in range(matches):
        if(trip.iat[i,-1]=='-1'):
            ender = str(-1)
        else:
            ender = trip.iat[i,4]            
        print(trip.iat[i,1]+' '+trip.iat[i,4]+' '+trip.iat[i,0]+' '+trip.iat[i,-1]+' '+ender+'\n')
    print(str(cost)+'\n')

file.close()

