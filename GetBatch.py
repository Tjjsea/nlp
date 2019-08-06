#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

class Batch():
    def __init__(self):
        self.word=[]
        self.char=[]
        self.tag=[]
        self.arc=[]
        self.label=[]

def getbatch(mode,batch_size=64):
    filepath='UD_English-EWT/en_ewt-ud-'+mode+'.conllu'
    data=open(filepath,encoding='utf-8').readlines()
    chardict=json.load('data/chat.json')
    worddict=json.load('data/words.json')
    batches=[]
    batch=Batch()
    for line in enumerate(data):
        line=line.strip()
        if not line:
            pass
        if line[0]=='#':
            continue
        word=line[1]
        tag=line[3]
        head=line[6]
        label=line[7]


if __name__=='__main__':
    batch_size=64
    trainpath='UD_English-EWT/en_ewt-ud-train.conllu'
    devpath='UD_English-EWT/en_ewt-ud-dev.conllu'
    testpath='UD_English-EWT/en_ewt-ud-test.conllu'

    trainfile=open(trainpath,encoding='utf-8').readlines()
