#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from collections import defaultdict, namedtuple

Arc = namedtuple('Arc', ('tail', 'weight', 'head'))

def max_spanning_arborescence(arcs, sink):
    good_arcs = [] #MST的弧
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[sink] = sink
    while True:
        min_arc_by_tail_rep = {} #以某个节点为tail的权值最大的弧 tail:arc
        successor_rep = {}       #保存上面记录的弧的tail和head，相当于GM
        for arc in arcs:         #对每个点，选择权重最大的入边
            if arc.tail == sink:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if tail_rep not in min_arc_by_tail_rep or min_arc_by_tail_rep[tail_rep].weight < arc.weight:
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, sink)
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, sink)
        good_arcs.extend(min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}


def find_cycle(successor, sink):
    '''
    寻找弧
    successor:GM,{tail:head}
    sink:根节点
    return:cycle,列表，含有环的序号
    '''
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None


def spanning_arborescence(arcs, sink):
    arcs_by_head = defaultdict(list)
    weights=0
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc.head #符合treebank格式
        weights+=arc.weight
        stack.extend(arcs_by_head[arc.tail])
    solution_arc_by_tail[sink] = sink #根节点没有head

    return [solution_arc_by_tail[k] for k in sorted(solution_arc_by_tail)],weights

def trans(G):
    allarcs=[]
    for b in range(G.shape[0]):
        arcs=[]
        for i in range(G.shape[1]):
            for j in range(G.shape[2]):
                if i==j:
                    continue
                arcs.append(Arc(j,G[b,i,j],i))
        allarcs.append(arcs)
    return allarcs #batches

def MST(G):
    '''
    获取最大生成树
    G:分数矩阵，ndarray
    return:msts:[batch_size,sequence_length] mweights:[batch_size],树的权重
    '''
    allarcs=trans(G)
    msts,mweights=[],[]
    for idx,arcs in enumerate(allarcs):
        mst,mweight={},-999
        for i in range(G.shape[1]):
            tree,weight=max_spanning_arborescence(arcs,i)
            if weight>mweight:
                mst,mweight=tree,weight  #mst:列表，表示对应词的head的序号，head为0表示无head
        msts.append(mst)
        mweights.append(mweight)
    msts=tf.cast(msts,tf.int32)
    mweights=tf.cast(mweights,tf.float32)
    return msts,mweights

def GetScore(G,heads):
    '''
    根据给定目标head，计算相应树的权重
    '''
    weights=[]
    for b in range(len(heads)):
        temp=0
        for i,h in enumerate(heads[b]):
            if i==h: #根节点
                continue
            temp+=G[b,h,i]
        weights.append(temp)
    weights=tf.cast(weights,tf.float32)
    return weights

if __name__=='__main__':
    G=[[[2,5,10,18,21],
       [3,12,9,18,20],
       [5,5,19,30,6],
       [15,18,22,8,9],
       [6,10,24,16,28]]]
    
    
    with tf.Session() as sess:
        G=tf.cast(G,tf.int32)
        msts,mweights=MST(G.eval())
        print(msts)
        print(sess.run(mweights))