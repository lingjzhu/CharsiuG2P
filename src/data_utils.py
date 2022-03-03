#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
from datasets import Dataset



def load_pronuncation_dictionary(language: str) -> Dataset:
    
    words = []
    prons = []
    variants = []
    with open(os.path.join('./dicts',language+'.tsv'),'r') as f:
        for line in f.readlines():
            word, pron = line.strip().split('\t')
            if ',' in pron:
                variant = ','.join(pron.split(',')[1:]).replace(' ','')
                pron = pron.split(',')[0]
            else:
                variant = ''
            words.append(word)
            prons.append(pron)
            variants.append(variant)
    
    data = pd.DataFrame()
    data['word'] = words
    data['pron'] = prons
    data['variant'] = variants
    data['language'] = language
    return Dataset.from_pandas(data)
            


if __name__ == "__main__":
    
    data = load_pronuncation_dictionary('yue')
