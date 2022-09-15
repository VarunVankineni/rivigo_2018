# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:52:14 2018

@author: Varun
"""

import pandas as pd
import numpy as np
import scipy
import os
from multiprocessing import Pool,Process,Queue,freeze_support
import time

    
def returnfunc(x,t):
    return x+t

if __name__ == '__main__':
    freeze_support()
    pooler = Pool(processes = 4)
    x = pooler.apply_async(returnfunc,args=(1,2,))
    #print(x.get())