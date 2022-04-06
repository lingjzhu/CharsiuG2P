#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch

import os
from datasets import load_metric
from dataclasses import dataclass
from typing import Union, Dict, List, Optional
from transformers import AdamW, AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers import (
    DataCollator,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    set_seed,
)

import sys
sys.path.append('src')
from data_utils import load_pronuncation_dictionary, load_all_pronuncation_dictionaries
from ByT5_MoE import SwitchT5ForConditionalGeneration


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
        pron_batch = self.tokenizer(prons,padding=self.padding,add_special_tokens=True,
                          return_attention_mask=True,return_tensors='pt')
        
        # replace padding with -100 to ignore loss correctly
        batch['labels'] = pron_batch['input_ids'].masked_fill(pron_batch.attention_mask.ne(1), -100)


        return batch
    
    

                


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer, 'wer':wer}



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--model',type=str,default='byt5',help='switch or byt5')
    parser.add_argument('--model_name',type=str,default='google/byt5-small')
    parser.add_argument('--train_data',type=str,default='data/train')
    parser.add_argument('--dev_data',type=str,default='data/dev')
    parser.add_argument('--test_data',type=str,default='data/test')
    parser.add_argument('--pretrained_model', type=bool, default=False)
    parser.add_argument('--output_dir',type=str,default=None)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',default=None,type=str)
    parser.add_argument('--resume_from_checkpoint',action='store_true')
    parser.add_argument('--language',default=None,type=str)
    
    
    # trainign hyperparameters
    parser.add_argument('--fp16',type=bool,default=False,help="fp16 not available for switch transformers")
    parser.add_argument('--train_batch_size',type=int,default=256)
    parser.add_argument('--learning_rate',type=float,default=3e-4)
    parser.add_argument('--warmup_steps',type=int,default=1000)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--eval_batch_size',type=int,default=512)
    parser.add_argument('--gradient_accumulation',type=int,default=2)
    parser.add_argument('--logging_steps',type=int,default=1000)
    parser.add_argument('--save_steps',type=int,default=5000)
    parser.add_argument('--eval_steps',type=int,default=5000)
    parser.add_argument('--unk_prob',type=float,default=0.85)
    
    # model hyperparameters
    parser.add_argument('--num_encoder_layers',type=int,default=6)
    parser.add_argument('--num_decoder_layers',type=int,default=2)
    parser.add_argument('--d_model',type=int,default=256)
    parser.add_argument('--d_kv',type=int,default=64)
    parser.add_argument('--d_ff',type=int,default=512)
    
    # MoE hyperparameters
    parser.add_argument('--capacity_factor',type=float,default=1.0)
    parser.add_argument('--n_experts',type=int,default=8)
    parser.add_argument('--load_balancing_loss_weight',type=float,default=1e-2)
    parser.add_argument('--is_scale_prob',type=bool,default=True)
    parser.add_argument('--drop_tokens',type=bool,default=False)
    
    args = parser.parse_args()
    
    # setting the evaluation metrics
    cer_metric = load_metric("cer")
    wer_metric = load_metric('wer')

    if args.train == True:
        
        # loading and preprocessing data
        if not args.language:
            train_data = load_all_pronuncation_dictionaries(args.train_data, prefix=True, mask_prob=args.unk_prob)
            train_data = train_data.map(prepare_dataset)    
            train_dataset = train_data

            dev_data = load_all_pronuncation_dictionaries(args.dev_data, prefix=True)
            dev_data = dev_data.map(prepare_dataset)    
            dev_dataset = dev_data  
            
        elif args.language:
            
            train_data = load_pronuncation_dictionary(path=args.train_data,language=args.language,prefix=True)
            train_data = train_data.map(prepare_dataset)    
            train_dataset = train_data

            dev_data = load_pronuncation_dictionary(path=args.dev_data,language=args.language,prefix=True)
            dev_data = dev_data.map(prepare_dataset)    
            dev_dataset = dev_data              


        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # intitalizing the model
        if args.pretrained_model == True:
            print('Loading pretrained model...')
            if args.model == 'byt5':
                model = T5ForConditionalGeneration.from_pretrained(args.model_name)
            elif args.model == 'switch':
                model = SwitchT5ForConditionalGeneration.from_pretrained(args.model_name)
        else:

            config = T5Config.from_pretrained(args.model_name)

            config.num_decoder_layers = args.num_decoder_layers
            config.num_layers = args.num_encoder_layers
            config.d_kv = args.d_kv
            config.d_model = args.d_model
            config.d_ff = args.d_ff

            if args.model == 'byt5':
                print('Initializing a ByT5 model...')
                model = T5ForConditionalGeneration(config)

            elif args.model == 'switch':
                print('Initialing Switch ByT5...')
                config.capacity_factor = args.capacity_factor
                config.is_scale_prob = args.is_scale_prob
                config.n_experts = args.n_experts
                config.drop_tokens = args.drop_tokens
                config.lambda_ = args.load_balancing_loss_weight
                model = SwitchT5ForConditionalGeneration(config)



        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            generation_num_beams=5,
            evaluation_strategy="steps",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type="cosine",
            fp16=args.fp16, 
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True
        )


        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )

        if args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
            
        trainer.save_model(args.output_dir)
    
    
    elif args.evaluate == True:
        
        if not args.language:
            test_data = load_all_pronuncation_dictionaries(args.test_data, prefix=True)
            test_data = test_data.map(prepare_dataset)    
            test_dataset = test_data    
            
        elif args.language:
            
            test_data = load_pronuncation_dictionary(path=args.test_data,language=args.language,prefix=True)
            test_data = test_data.map(prepare_dataset)    
            test_dataset = test_data    
           

        
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        if args.model == 'byt5':
            model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
        elif args.model == 'switch':
            model = SwitchT5ForConditionalGeneration.from_pretrained(args.checkpoint)
            
            
            
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            generation_num_beams=5,
            evaluation_strategy="steps",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=2
        )
        
        
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )
            
            
        
        eval_results = trainer.evaluate(eval_dataset=test_dataset,num_beams=5)
        print(eval_results)
        with open(os.path.join(args.output_dir, 'results'),'w') as out:
            out.write('%s\t%s\t%s\n'%(args.language,eval_results['eval_cer'],eval_results['eval_wer']))