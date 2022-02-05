#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from transformers import AdamW
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class DataCollatorWithPadding:

    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        inputs = [{"word": feature["word"]} for feature in features]
        prons = [{"pronunciation": feature["pronunciation"]} for feature in features]

        batch = tokenizer(words,padding=padding,add_special_tokens=False,
                          return_attention_mask=True,return_tensors='pt')
        pron_batch = tokenizer(prons,padding=padding,add_special_tokens=False,
                          return_attention_mask=True,return_tensors='pt')
        
        # replace padding with -100 to ignore loss correctly
        batch['labels'] = pron_batch['input_ids'].masked_fill(pron_batch.attention_mask.ne(1), -100)


        return batch


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    
    args = parser.parse_args()