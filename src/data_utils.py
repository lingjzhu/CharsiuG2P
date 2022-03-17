#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm import tqdm


def load_pronuncation_dictionary(path: str) -> Dataset:
    
    words = []
    prons = []
    variants = []
    with open(os.path.join(path,language+'.tsv'),'r') as f:
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
            


def load_all_pronuncation_dictionaries(path: str, prefix: bool = False, mask_prob: float = 0.0, lower_case: bool = True) -> Dataset:
    
    
    files = os.listdir(path)
    
    all_data = pd.DataFrame()
    
    for file in tqdm(files):
        words = []
        prons = []
        variants = []
        language = file.replace('.tsv','')
        with open(os.path.join(path,file),'r') as f:
            for line in f.readlines():
                word, pron = line.strip().split('\t')
                if ',' in pron:
                    variant = ','.join(pron.split(',')[1:]).replace(' ','')
                    pron = pron.split(',')[0]
                else:
                    variant = ''
                if prefix == True:
                    if mask_prob == 0.0:
                        word = '<'+language+'>:' + word
                    else:
                        if np.random.uniform() > mask_prob:
                            word = '<unk>:' + word
                        else:
                            word = '<'+language+'>:' + word
                words.append(word)
                prons.append(pron)
                variants.append(variant)
        
        data = pd.DataFrame()
        data['word'] = words
        data['pron'] = prons
        data['variant'] = variants
        data['language'] = language
        
        all_data = pd.concat([all_data,data],ignore_index=True)
    return Dataset.from_pandas(all_data)


if __name__ == "__main__":
    
    data = load_pronuncation_dictionary('yue')

    data = load_all_pronuncation_dictionaries('dicts')
