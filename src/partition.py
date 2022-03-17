#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from sklearn.utils import shuffle



dicts = os.listdir("dicts")


for d in dicts:
    
    entries = []
    with open(os.path.join('dicts',d),'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            entries.append((line[0],line[1].replace(' ','')))
            
    entries = shuffle(entries)
    
    
    test = entries[:500]
    dev = entries[500:550]
    train = entries[550:]
    
    with open(os.path.join('data/train',d),'w') as out:
        for w,p in train:
            out.write('%s\t%s\n'%(w,p))
            
    with open(os.path.join('data/dev',d),'w') as out:
        for w,p in dev:
            out.write('%s\t%s\n'%(w,p))
            
    with open(os.path.join('data/test',d),'w') as out:
        for w,p in test:
            out.write('%s\t%s\n'%(w,p))
            
    