#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

def extract(mode):
    '''
    从treebank中抽取数据，选取长度在4-36之间的句子
    '''
    filepath='UD_English-EWT/en_ewt-ud-'+mode+'.conllu'
    outpath='data/'+mode+'.json'
    datas=open(filepath,encoding='utf-8').readlines()
    exdatas=[]
    sentence={}
    count=0
    form,lemma,POStag,head,label=[],[],[],[],[]
    
    for line in datas:
        line=line.strip()
        if not line:
            if count<=3 or count>=37:
                count=0
                form,lemma,POStag,head,label=[],[],[],[],[]
                continue
            count=0
            sentence['word']=form
            sentence['lemma']=lemma
            sentence['POStag']=POStag
            sentence['head']=head
            sentence['label']=label
            exdatas.append(sentence)
            sentence={}
            form,lemma,POStag,head,label=[],[],[],[],[]
            continue
        if line[0]=='#':
            continue
        line=line.split('\t')
        if line[0].isdigit()==False:
            continue        
        form.append(line[1])
        lemma.append(line[2])
        POStag.append(line[3])
        try:
            line[6]=int(line[6])
            line[0]=int(line[0])
        except:
            print(line)
        
        if line[6]==0:
            line[6]=line[0]-1
        else:
            line[6]-=1
        head.append(line[6])
        la=line[7].find(':')
        if la!=-1:
            line[7]=line[7][:la]            
        label.append(line[7])
        count+=1
    with open(outpath,'w',encoding='utf-8') as fout:
        json.dump(exdatas,fout)

def GetWordict():
    filepath='UD_English-EWT/en_ewt-ud-train.conllu'
    datas=open(filepath,encoding='utf-8').readlines()
    wordict={"UNK":0,"PAD":1,"NUM":2}
    count=3
    for line in datas:
        line=line.strip()
        if not line:
            continue
        if line[0]=='#':
            continue
        line=line.split('\t')
        word=line[1]
        if word.isdigit():
            continue
        if word not in wordict:
            wordict[word]=count
            count+=1
    outpath='data/words.json'
    json.dump(wordict,open(outpath,'w'))

def GetChardict():
    char={'un':0,"pd":1}
    A=65
    a=97
    for i in range(26):
        char[chr(A+i)]=i+2
    for i in range(26):
        char[chr(a+i)]=i+28
    with open('data/char.json','w',encoding='utf-8') as fout:
        json.dump(char,fout)

extract('train')
extract('dev')
extract('test')