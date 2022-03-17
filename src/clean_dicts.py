#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:53:45 2022

@author: lukeum
"""


import re
import os
from tqdm import tqdm

path = 'original_dicts'

dicts = os.listdir(path)


for d in tqdm(dicts):
    
    entries = []
    with open(os.path.join(path,d),'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line)==2:
                entries.append((line[0],line[1].replace(' ','')))
            

    modified = []
    for w,p in entries:
        if not re.search('[\(\)]',p):
            modified.append((w,p))
            

    with open(os.path.join('dicts',d),'w') as out:
        for w,p in modified:
            out.write("%s\t%s\n"%(w,p))