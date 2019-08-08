#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

Maxlength=36
Charlength=16

class Batch():
    def __init__(self):
        self.word=[]     #[batdh_size,sequence_length]
        self.char=[]     #[batch_size,sequence_length,charlength]
        self.tag=[]      #[batch_size,sequence_length,num_tag]
        self.arc=[]      #[batch_size,sequence_length]
        self.label=[]    #[batch_size,sequence_length,num_label]
        self.position=[] #[batch_size,sequence_length]

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
            if len(pos)>Maxlength:
                pos=pos[:Maxlength]
            elif len(pos)<Maxlength:
                pos.extend(['X']*(Maxlength-len(pos)))
            num_pos=len(posdict)
            ttemp=[]
            for p in pos:
                tags=[0]*num_pos
                tags[posdict[p]]=1
                ttemp.append(tags)
            batch.tag.append(ttemp)
            
            head=st['head']
            if len(head)>Maxlength:
                head=head[:Maxlength]
            elif len(head)<Maxlength:
                head.extend([0]*(Maxlength-len(head)))
            batch.arc.append(head)

            label=st['label']
            if len(label)>Maxlength:
                label=label[:Maxlength]
            elif len(label)<Maxlength:
                label.extend(['PD']*(Maxlength-len(label)))
            num_label=len(labeldict)
            ltemp=[]
            for l in label:
                labels=[0]*num_label
                labels[labeldict[l]]=1
                ltemp.append(labels)
            batch.label.append(ltemp)

            #ptemp=list(range(Maxlength))
            ptemp=[[i] for i in range(Maxlength)]
            batch.position.append(ptemp)
        batches.append(batch)
    return batches         

if __name__=='__main__':
    batch_size=64
    trainpath='UD_English-EWT/en_ewt-ud-train.conllu'
    devpath='UD_English-EWT/en_ewt-ud-dev.conllu'
    testpath='UD_English-EWT/en_ewt-ud-test.conllu'

    batches=getbatch('train',batch_size)
    batch=batches[1]
    bd={}
    bd['word']=batch.word
    bd['char']=batch.char
    bd['tag']=batch.tag
    bd['arc']=batch.arc
    bd['label']=batch.label
    bd['position']=batch.position
    print(len(batch.word))
    #with open('batch.json','w') as fout:
    #    json.dump(bd,fout)
    #print(batches[0].arc)
    #print(batches[0].word)
