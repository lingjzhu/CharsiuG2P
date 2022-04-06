import argparse
import torch
import pandas as pd
import os
from datasets import load_metric
from dataclasses import dataclass
from tqdm import tqdm
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



def compute_metrics(labels_ids,pred_ids,tokenizer):

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer, 'wer':wer}

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

def prepare_dataset(batch):
    
    batch['input_ids'] = batch['word']
    batch['labels'] = batch['pron']
    
    return batch
    
    

if __name__ == "__main__":    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='switch',help='switch or byt5')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--test_data',type=str,default='data/test')
    parser.add_argument('--output',type=str,default=None)
    parser.add_argument('--checkpoint',default=None,type=str)
    parser.add_argument('--zero_shot',action='store_true')
    args = parser.parse_args()

    device = args.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if args.model == 'switch':
        model = SwitchT5ForConditionalGeneration.from_pretrained(args.checkpoint)
    elif args.model == 'byt5':
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)

    model.eval()
    model.to(device)

    # setting the evaluation metrics
    cer_metric = load_metric("cer")
    wer_metric = load_metric('wer')

    data = [i for i in os.listdir(args.test_data) if i.endswith('.tsv')]



    for lang in tqdm(data):
        language = lang.replace('.tsv','')
        if not args.zero_shot:
            ind_data = load_pronuncation_dictionary(path=os.path.join(args.test_data,lang),language=language,prefix=True)
        elif args.zero_shot:
            ind_data = load_pronuncation_dictionary(path=os.path.join(args.test_data,lang),language='<unk>:',prefix=True)
        ind_data = ind_data.map(prepare_dataset)    

        collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=False)
        loader = torch.utils.data.DataLoader(ind_data,collate_fn=collator)

        preds = []
        labels = []
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            label = batch['labels'].squeeze()
            labels.append(label)
            with torch.no_grad():
                pred = model.generate(input_ids,num_beams=5).squeeze().cpu()
            preds.append(pred)
        metrics = compute_metrics(labels,preds,tokenizer)
        with open(args.output,'a') as out:
            out.write('%s\t%s\t%s\n'%(language,metrics['cer'],metrics['wer']))

