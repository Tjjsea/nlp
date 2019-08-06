#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

length=0
with open('UD_English-EWT/en_ewt-ud-train.conllu',encoding='utf-8') as fin:
    datas=fin.readlines()
    for line in datas:
        line=line.strip()
        if not line:
            continue
        if line[0]=='#':
            continue
        line=line.split('\t')
        if not line[0].isdigit():
            continue
        if int(line[0])>length:
            length=int(line[0])
    print(length)