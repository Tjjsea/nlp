#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

NE=-1

def iscycle(v,recStack,G,cycle):
    recStack[v]=True   #该点被访问
    cycle.append(v)
    for node in range(len(G[v])):
        if G[v][node] == NE or v == node:
            continue
        if recStack[node]==False:
            if iscycle(node,recStack,G,cycle):
                return True
        else:
            cycle.append(node)
            return True
    
    recStack[v]=False
    cycle=[]
    return False

def detect_cycle(G):
    '''
    寻找有向图中的环，若存在环，返回环中的点；否则，返回空列表
    '''
    recStack=[False]*len(G)  #记录某个点是否被访问过
    cycle=[]
    for node in range(len(G)):
        if iscycle(node,recStack,G,cycle):
            return cycle[cycle.index(cycle[-1]):-1]
    return cycle

def CLE(G):
    '''
    Chu-Liu-Edmonds algorithm
    '''
    NewG=[]
    for j in range(len(G[0])):
        allv=list(G[:,j])
        allv[j]=NE
        NewG.append(allv.index(max(allv)))
    GM=[[NE]*len(G) for _ in range(len(G))]
    for i,e in enumerate(NewG):
        GM[e][i]=G[e][i]
    cycle=detect_cycle(GM)
    if not cycle:
        return np.array(GM)
    
    Gc=contract(G,cycle)
    y=CLE(Gc)
    
    
def contract(G,C):
    pass
    

if __name__=='__main__':
    G=[[2,5,10,18,21],
       [3,12,9,18,20],
       [5,5,19,30,6],
       [15,18,22,8,9],
       [6,10,24,16,28]]
    NG=[[NE,NE,NE,3,NE],
        [NE,NE,2,NE,NE],
        [NE,NE,NE,NE,4],
        [NE,1,NE,NE,NE],
        [NE,1,NE,NE,NE]]
    G=np.array(G)
    NG=np.array(NG)
    CLE(G)