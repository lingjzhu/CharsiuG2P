#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from jiwer import wer,cer
import numpy as np
import torch


from datasets import load_metric
from dataclasses import dataclass, field
from typing import Union, Dict, List, Optional
from transformers import AdamW, AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers import (
    HfArgumentParser,
    DataCollator,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    set_seed,
)

import sys
sys.path.append('src')
from data_utils import load_pronuncation_dictionary


def prepare_dataset(batch):
    
    batch['input_ids'] = batch['word']
    batch['labels'] = batch['pron']
    
    return batch
    

@dataclass
class DataCollatorWithPadding:

    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        words = [feature["input_ids"] for feature in features]
        prons = [feature["labels"] for feature in features]

        batch = self.tokenizer(words,padding=self.padding,add_special_tokens=False,
                          return_attention_mask=True,return_tensors='pt')
        pron_batch = self.tokenizer(prons,padding=self.padding,add_special_tokens=False,
                          return_attention_mask=True,return_tensors='pt')
        
        # replace padding with -100 to ignore loss correctly
        batch['labels'] = pron_batch['input_ids'].masked_fill(pron_batch.attention_mask.ne(1), -100)


        return batch
    
    

                
                
def evaluate_all_metrics(model,dataset,args):
    
    model.eval()
    words = tokenizer('힐끔힐끔', return_tensors="pt")
    out = model.generate(**words.to(device),num_beams=5).squeeze()
    tokenizer.decode(out,skip_special_tokens=True)
    return 
        


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}



if __name__ == "__main__":
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('grad_acc', type=int, default=8,
                        help='an integer for the accumulator')    
    args = parser.parse_args()
    '''
    
    cer_metric = load_metric("cer")
    '''
    hypotheses = ['ki:m˥','a:m˥']
    ground_truth = ['hi:m˥','a:m˥']
    wer(ground_truth,hypotheses)
    cer(ground_truth,hypotheses)
    '''
    
    
    
    data = load_pronuncation_dictionary('other_dicts/fr-fr')
    data = data.map(prepare_dataset)    
    train_dataset = data
    eval_dataset = data
    
    config = T5Config.from_pretrained('google/byt5-small')
    #model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
    
    
    config
    config.num_decoder_layers = 2
    config.num_layers = 8
    config.d_kv = 32
    config.d_model = 256
    config.d_ff = 512

    model = T5ForConditionalGeneration(config)
    
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=512,
        fp16=False, 
        output_dir="./results",
        logging_steps=2,
        save_steps=1000,
        eval_steps=1000,
    )
    
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()