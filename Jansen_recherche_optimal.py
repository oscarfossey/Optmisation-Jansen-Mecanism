# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:43:56 2020

@author: admin
"""


import time 
import Jansen_global as Jgbl
import algo_optimisation as opt
import numpy as np
import statistics as stat

L = 10
l = 15
start = time.time()
n_test_MC = 15
intervalles = [[1.1,1.5],[3,4],[3.7,4.5],[3.5,4.4],[3.7,4.3],[5,6],[3.5,4.5],[3.7,4.2],[6.1,7],[4.5,5.2],[4.7,5.4],[5.8,6.4],[0.65,1.3]]

def score_algo_MC(Lnominales):
    
    Scoresx_MC = []
    Scoresy_MC = []
    for i in range(n_test_MC):
        LnominalesArray = np.array(Lnominales)
        S = Jgbl.Simulation(LnominalesArray,Jgbl.Ecartype,Jgbl.L3p8,Jgbl.RotationIncrements)
        Scoresx_MC.append(S[0])
        Scoresy_MC.append(S[1])
    
    return [180/np.pi*np.arcsin(stat.mean(Scoresx_MC)/L),180/np.pi*np.arcsin(stat.mean(Scoresy_MC)/l)]



# A = opt.minimize(intervalles,score_algo_MC,[0])
# print(A)
# print("temps d'éxectution " + str(time.time()-start))
