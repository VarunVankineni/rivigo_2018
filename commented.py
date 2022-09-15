# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:22:41 2018

@author: Varun
"""

import pandas as pd
import numpy as np
import scipy
import time
from pulp import LpProblem,LpMinimize,LpVariable,LpStatus,LpBinary,LpSolverDefault,value


"""
Helper functions to calculate waiting cost function and 
smallest journey time between two pitstops
"""
def waitCost(a):
    if(a>10):
        return (2*a)-10
    elif(a>=0):
        return a
    else:
        return 0
    
def quickJourney(x):
    timer = 0
    a = x[0]
    b = x[1]
    sign = np.sign(ord(b)-ord(a))
    while(a!=b):
        nextstop = chr(ord(a)+sign) 
        timer+=pit.loc[a,nextstop]
        a = nextstop
    return timer

"""
Read input file from text document and write output to another
"""
file = open('E:/Acads/9th sem/rivigo/2b,c/input.txt', 'rb')
wfile = open("outputt.txt","w")
N = int(file.readline())
wfile.write(str(N)+'\n') #write number of cases to output file


for i in range(1):#looping over all test cases 
    #read current case and lines of input
    currentcase,lines = file.readline().decode("utf-8").rstrip().split(" ")
    #read list of pit stops
    pitlist = file.readline().decode("utf-8").rstrip().split(" ")
    #read parameters of the current case
    [t,s,ow,r] = [int(x) for x in file.readline().decode("utf-8").rstrip().split(" ")]
    




