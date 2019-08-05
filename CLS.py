#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf

def iscycle(v,recStack,G):
    recStack[v]=True
    for node in range(len(G[v])):
        if G[v][node] == -1 or v == node:
            continue
        if recStack[node]==False:
            if iscycle(node,recStack,G):
                return True
        else:
            return True
    
    recStack[v]=False
    return False

def detect_cycle(G):
    recStack=[False]*len(G)
    for node in range(len(G)):
        if iscycle(node,recStack,G):
            return True
    return False

def CLS(G):
    '''
    Chu-Liu-Edmonds algorithm
    '''
    pass

if __name__=='__main__':
    G=[[2,5,10,18,21],
       [3,12,9,18,20],
       [-1,5,19,30,6],
       [15,18,22,8,9],
       [6,10,24,16,28]]
    NG=[[-1,-1,-1,3,-1],
        [-1,-1,2,-1,-1],
        [-1,-1,-1,-1,4],
        [-1,1,-1,-1,-1],
        [-1,-1,-1,-1,-1]]
    print(detect_cycle(NG))
