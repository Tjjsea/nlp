#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

Maxlength=36
Charlength=16

class Batch():
    def __init__(self):
        self.word=[]
        self.char=[]
        self.tag=[]
        self.arc=[]
        self.label=[]

def getbatch(mode,batch_size=64):
    filepath='data/'+mode+'.json'
    data=json.load(open(filepath,encoding='utf-8'))
    chardict=json.load(open('data/char.json'))
    worddict=json.load(open('data/words.json'))
    posdict=json.load(open('data/POStags.json'))
    labeldict=json.load(open('data/arclabel.json'))
    batches=[]
    for i in range(0,len(data),batch_size):
        part=data[i:min(len(data),i+batch_size)]
        batch=Batch()
        for st in part:            
            words=st['word']
            if len(words)>Maxlength:
                words=words[:Maxlength]
            elif len(words)<Maxlength:
                words.extend(['PAD']*(Maxlength-len(words))) 
            wtemp,ctemp=[],[]           
            for word in words:
                if word.isdigit():
                    word="NUM"
                wtemp.append(worddict.get(word,worddict['UNK']))

                char=[]
                if len(word)>Charlength:
                    word=word[:Charlength]
                for c in word:
                    char.append(chardict.get(c,chardict['un']))
                if len(char)<Charlength:
                    char.extend([chardict['pd']]*(Charlength-len(char)))
                ctemp.append(char)
            batch.word.append(wtemp)
            batch.char.append(ctemp)
            
            pos=st['POStag']
            num_pos=len(posdict)
            ttemp=[]
            for p in pos:
                tags=[0]*num_pos
                tags[posdict[p]]=1
                ttemp.append(tags)
            batch.tag.append(ttemp)
            
            batch.arc.append(st['head'])

            label=st['label']
            num_label=len(labeldict)
            ltemp=[]
            for l in label:
                labels=[0]*num_label
                labels[labeldict[l]]=1
                ltemp.append(labels)
            batch.label.append(ltemp)
        batches.append(batch)
    return batches         

if __name__=='__main__':
    batch_size=64
    trainpath='UD_English-EWT/en_ewt-ud-train.conllu'
    devpath='UD_English-EWT/en_ewt-ud-dev.conllu'
    testpath='UD_English-EWT/en_ewt-ud-test.conllu'

    batches=getbatch('train')
    print(batches[0].arc)
    print(batches[0].word)
