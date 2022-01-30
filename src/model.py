#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 09:46:13 2022

@author: lukeum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from transformers import T5ForConditionalGeneration,AutoTokenizer, AdamW, T5Config
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')


data = []
with open('ko.txt') as f:
    for line in f.readlines():
        word, pron = line.strip().split('\t')
        if ',' in pron:
            pron = pron.split(',')[0]
        data.append((word,pron.replace('/','')))
        
        
def batchify(data,batch_size=32):
    
    size = len(data)
    for idx in range(0,size,batch_size):
        words = [i[0] for i in data[idx:min(size,idx+batch_size)]]
        prons = [i[1] for i in data[idx:min(size,idx+batch_size)]]
        
        batch = tokenizer(words,padding=True,add_special_tokens=False,
                          return_attention_mask=True,return_tensors='pt')
        pron_batch = tokenizer(prons,padding=True,add_special_tokens=False,
                          return_attention_mask=True,return_tensors='pt')
        batch['labels'] = pron_batch['input_ids']
        yield batch
        
        
config = T5Config.from_pretrained('google/byt5-small')
#model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')


config
config.num_decoder_layers = 2
config.num_layers = 8
config.d_kv = 32
config.d_model = 256
config.d_ff = 512

model = T5ForConditionalGeneration(config)


epoch = 5
grad_acc = 1
lr = 1e-4
device = 'cuda'


model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=lr,weight_decay=1e-4)

losses = []
for _ in range(epoch):
    data = shuffle(data)
    for i, batch in tqdm(enumerate(batchify(data,batch_size=32))):
    
        loss = model(**batch.to(device)).loss
        
        loss.backward()
        if i%grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
        if i%100 == 0:
            print("Loss: %s at Iteration: %i"%(loss.item(),i))
            losses.append(loss.item())
            
            
model.eval()
words = tokenizer('힐끔힐끔', return_tensors="pt")
out = model.generate(**words.to(device),num_beams=5).squeeze()
tokenizer.decode(out,skip_special_tokens=True)


if __name__ == "__main__":
    pass