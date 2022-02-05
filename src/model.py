#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from transformers import T5ForConditionalGeneration,AutoTokenizer, AdamW, T5Config
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')




        
        
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