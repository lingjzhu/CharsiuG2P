# CharsiuG2P

### Introduction
CharsiuG2P is transformer based tool for grapheme-to-phoneme conversion in 100 languages. Given an orthographic word, CharsiuG2P predicts its pronunciation through a neural G2P model.

This repository also contains instructions to replicate our Interspeech 2022 paper *ByT5 model for massively multilingual grapheme-to-phoneme conversion* [[arXiv]](https://arxiv.org/abs/2204.03067) [[pdf]](https://arxiv.org/pdf/2204.03067.pdf).  

### Table of content
- [Introduction](https://github.com/lingjzhu/CharsiuG2P#introduction)
- [Usage](https://github.com/lingjzhu/CharsiuG2P#usage)
- [Results](https://github.com/lingjzhu/CharsiuG2P#results)
- [Pretrained models](https://github.com/lingjzhu/CharsiuG2P/blob/main/README.md#pretrained-models)
- [Training and fine-tuning](https://github.com/lingjzhu/CharsiuG2P#training-and-fine-tuning)
- [Evaluation](https://github.com/lingjzhu/CharsiuG2P#evaluation)
- [G2P Datasets](https://github.com/lingjzhu/CharsiuG2P#g2p-datasets)
- [Docker image for *espeak-ng* ](https://github.com/lingjzhu/CharsiuG2P#docker-image-for-espeak-ng)
- [Disclaimer](https://github.com/lingjzhu/CharsiuG2P/blob/main/README.md#disclaimer)
- [Contact](https://github.com/lingjzhu/CharsiuG2P/blob/main/README.md#contact)


### Usage

The model can be directly loaded from Huggingface Hub. Note that this model assume that input words are already tokenized into individual words. 
- For languages such as Chinese, Korean, Japanese (CJK languages) and some southeast Asian languages, words are not separated by spaces. An external tokenizers must be used before feeding words into this model.
- Each word must be proceeded by a language code prefix, which is based on ISO-639 with some slight modification to distinguish local dialects/variants. For example, the prefix code for American English is '\<eng-us\>: ' (**The space following the colon cannot be omitted!**). The full list of language codes can be found in this [document](https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit#gid=557940309). 
- For the sake of convenience, it is suggested that the .generate function is used to handle outputs. However, this could slows down the inference time significantly. 
- We accidentally left out Korean in our original model (sorry!). Updated models that include Korean has been uploaded to Huggingface Hub. The following are the updated models that work also on Korean. 
  - `charsiu/g2p_multilingual_byT5_tiny_8_layers_100`
  - `charsiu/g2p_multilingual_byT5_tiny_12_layers_100`
  - `charsiu/g2p_multilingual_byT5_tiny_16_layers_100`
  - `charsiu/g2p_multilingual_byT5_small_100`
```
from transformers import T5ForConditionalGeneration, AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

# tokenized English words
words = ['Char', 'siu', 'is', 'a', 'Cantonese', 'style', 'of', 'barbecued', 'pork']
words = ['<eng-us>: '+i for i in words]

out = tokenizer(words,padding=True,add_special_tokens=False,return_tensors='pt')

preds = model.generate(**out,num_beams=1) # We do not find beam search helpful. Greedy decoding is enough. 
phones = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)
print(phones)
# Output: ['ˈtʃɑɹ', 'ˈsiw', 'ˈɪs', 'ˈɑ', 'ˈkæntəˌniz', 'ˈstaɪɫ', 'ˈɑf', 'ˈbɑɹbɪkˌjud', 'ˈpɔɹk']
```

```
# tokenized Thai words
words = ['<tha>: ภาษา', '<tha>: ไทย']
out = tokenizer(words,padding=True,add_special_tokens=False,return_tensors='pt')
preds = model.generate(**out,num_beams=1)
phones = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)
print(phones)
# Output: ['pʰaː˧.saː˩˩˦', 'tʰaj˧']
# Correct pronunciation on wikipedia: [pʰāːsǎːtʰāj]
```

### Results
Results for different models are available at [multilingual_results/](https://github.com/lingjzhu/CharsiuG2P/tree/main/multilingual_results).
The format is language PER WER. 

### Pretrained models
Pretrained models are hosted at [HuggingFace model hub](https://huggingface.co/charsiu) with the prefix "G2P". Multilingual models were uploaded. We are still trying to figure out how to host 100 monolingual models.

### Training and fine-tuning

Here we provide the code for training and fine-tuning the ByT5 G2P model. 

In addition to ByT5, we also included a Switch ByT5 model class, which is essentially a [switch transformer](https://arxiv.org/pdf/2101.03961.pdf) that takes byte-leve inputs. While a sparse transformer can theoretically increase parameters without increasing computational costs. We did not find it much faster that the vanilla ByT5 model. It could be that switch transformers are not beneficial at our scale (small model and small datasets). While the extensive results for switch ByT5 models are not included in our paper, we still make the code, pretrained models and the results available, in the hope that someone might find them helpful.

**Note**. The code we used to train and finetune models in our paper can be found in [notebooks/](https://github.com/lingjzhu/CharsiuG2P/tree/main/notebooks) and [train.py](https://github.com/lingjzhu/CharsiuG2P/blob/main/src/train.py).


Finetune a pretrained ByT5 on all languages.
```
python src/train.py --output_dir path_to_output --pretrained_model True --train --train_batch_size 64 --gradient_accumulation 8 --eval_batch_size 128 
```

Train a 8-layer ByT5 with randomly initalize weights on all languages.
```
python src/train.py --output_dir path_to_output --num_encoder_layers 8 --num_decoder_layers 4 --d_ff 1024  --model byt5  
```

Train a 8-layer mT5 model with 128 hidden dimensions and a feedforward layer of 256 dimensions on all languages.
```
python src/train.py --output_dir path_to_output --num_encoder_layers 4 --num_decoder_layers 4  --model byt5 --model_name google/mt5-small --train --train_batch_size 64 --gradient_accumulation 4 --d_model 128 --d_ff 256 --eval_batch_size 128
```
Train a 6-layer Switch ByT5 with 64 experts on all languages.
```
python src/train.py --output_dir path_to_output --train --switch --num_encoder_layers 4 --n_experts 64 --num_decoder_layers 2
```

Finetune a ByT5 model on a single language.
```
!python src/train.py --output_dir path_to_output --language dsb --pretrained_model True --train --train_batch_size 32  --gradient_accumulation 1 --eval_batch_size 64 --train_data data/low_resource/train/dsb.tsv --dev_data data/low_resource/dev/dsb.tsv --model_name pretrained_model_path --learning_rate 1e-4 --save_steps 100 --logging_steps 50 --eval_steps 100 --epochs 50
```



### Evaluation

You can evaluate our model using the following command lines. 

Evaluate a multilingual ByT5 model on all languages.
```
python src/train.py --checkpoint path_to_pretrained_model_checkpoint  --evaluate --model byt5 --output_dir path_to_output
```

Evaluate a ByT5 model on a single language.
```
python src/train.py --checkpoint path_to_pretrained_model_checkpoint --language dsb --evaluate --model byt5 --test_data data/low_resource/test/dsb.tsv --output_dir path_to_output
```

### G2P Datasets
A detailed catalogue of pronunciation dictionaries with downloadable links can be found in [this form](https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?usp=sharing).  

We also make detailed documentation of the sources and the license of these data in *merge_final* page of the above form.  The sources of the pronunciationaries are in [dicts/](https://github.com/lingjzhu/CharsiuG2P/tree/main/dicts). The train/dev/test splits are in [data/](https://github.com/lingjzhu/CharsiuG2P/tree/main/data). **Please cite both our article and the original sources to acknowledge the original authors if you use the data.**

All data we collected are in [sources/](https://github.com/lingjzhu/CharsiuG2P/tree/main/sources). The source and license information for each file is available in [sources/info](https://github.com/lingjzhu/CharsiuG2P/tree/main/sources/info).


Almost all of the data here come with licenses that allow redistribution. For the rest of them, the license is unspecified. **If you are one of the creators of these data and do not wish us to host them, please let us know and we will immediately remove them per your request.** 

#### Attribution and Citation

Please cite our article:  
```
@article{zhu2022charsiu-g2p,
  title={ByT5 model for massively multilingual grapheme-to-phoneme conversion},
  author={Zhu, Jian and Zhang, Cong and Jurgens, David},
  url = {https://arxiv.org/abs/2204.03067},
  doi = {10.48550/ARXIV.2204.03067},
  year={2022}
 }
```
or

```
J. Zhu, C. Zhang, and D. Jurgens, “Byt5 model for massively
multilingual grapheme-to-phoneme conversion,” 2022. [Online]. Available:
https://arxiv.org/abs/2204.03067  
```

**The resources we collected include:**  

WikiPron (multiple languages):  

```
@inproceedings{lee-etal-2020-massively,
    title = "Massively Multilingual Pronunciation Modeling with {W}iki{P}ron",
    author = "Lee, Jackson L.  and
      Ashby, Lucas F.E.  and
      Garza, M. Elizabeth  and
      Lee-Sikka, Yeonju  and
      Miller, Sean  and
      Wong, Alan  and
      McCarthy, Arya D.  and
      Gorman, Kyle",
    booktitle = "Proceedings of LREC",
    year = "2020",
    publisher = "European Language Resources Association",
    pages = "4223--4228",
}
```

eSpeak NG (multiple languages). Word lists for some languages are acquired via [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download).   

```
@misc{espeakng,
  title = {{eSpeak NG}},
  year = {2022},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/espeak-ng/espeak-ng}},
}

@inproceedings{goldhahn2012building,
  title={Building large monolingual dictionaries at the leipzig corpora collection: From 100 to 200 languages},
  author={Goldhahn, Dirk and Eckart, Thomas and Quasthoff, Uwe},
  booktitle={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
  pages={759--765},
  year={2012}
}
```

ipa-dict (multiple languages):  

```
@misc{ipa-dict,
  title = {{ipa-dict}},
  year = {2020},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/open-dict-data/ipa-dic}},
}
```

Kurdish (kur):  

```
@article{veisi2020toward,
  title={Toward Kurdish language processing: Experiments in collecting and processing the AsoSoft text corpus},
  author={Veisi, Hadi and MohammadAmini, Mohammad and Hosseini, Hawre},
  journal={Digital Scholarship in the Humanities},
  volume={35},
  number={1},
  pages={176-193},
  year={2020},
  publisher={Oxford University Press}
}

@article{ahmadi2019rule,
  title={A Rule-Based Kurdish Text Transliteration System},
  author={Ahmadi, Sina},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)},
  volume={18},
  number={2},
  pages={18},
  year={2019},
  publisher={ACM}
}
```

Britfone (eng-uk):  

```
@misc{britfone,
  title = {{Britfone}},
  author = {Llarena, Jose},
  year = {2017},
  journal = {GitHub repository},
  howpublished =  {\url{https://github.com/JoseLlarena/Britfone}},
}
```

Thai (tha):  

```
@misc{thai-g2p,
  title = {{thai-g2p}},
  author = {Phatthiyaphaibun, Wannaphong},
  year = {2020},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sigmorphon/2020/tree/master/task1/}},
}
```

Spanish (spa-latin):  

```
@misc{sandiago-spanish,
  title = {{Santiago Spanish Lexicon
}},
  author = {Morgan, John},
  year = {2017},
  journal = {GitHub repository},
  howpublished = {\url{https://www.openslr.org/34/}},
}
```

Swedish (swe):  

```
@misc{Sprakbanken_Swe,
  title = {{Sprakbanken Swedish pronunciation dictionary}},
  author = {Phatthiyaphaibun, Wannaphong},
  year = {2020},
  journal = {GitHub repository},
  howpublished = {\url{https://www.openslr.org/29/}},
}
```





 - A collection of publicly available G2P data are listed below.
   - [British English (RP/Standard Southern British ) pronunciation dictionary](https://github.com/JoseLlarena/Britfone)
   - [British English pronunciations](https://www.openslr.org/14/)
   - [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
   - [G2P for (almost) any languages](https://drive.google.com/drive/u/0/folders/0B7R_gATfZJ2aWkpSWHpXUklWUmM?resourcekey=0-aj4VU-D4RztBPCFLKNNThQ)
   - [Kurdish G2P](https://github.com/AsoSoft/Kurdish-G2P-dataset)
   - [ipa-dict - Monolingual wordlists with pronunciation information in IPA](https://github.com/open-dict-data/ipa-dict#languages)
   - [Mandarin G2p](https://github.com/kakaobrain/g2pM)
   - [mg2p](https://github.com/bpopeters/mg2p)
   - [Santiago Spanish Lexicon](https://www.openslr.org/34/)
   - [Sigmorphon Multilingual G2P](https://github.com/sigmorphon/2020/tree/master/task1)
   - [Swedish pronunciation dictionary](https://www.openslr.org/29/)
   - [Thai G2P](https://github.com/wannaphong/thai-g2p/blob/master/wiktionary-11-2-2020.tsv)
   - [wiki-pronunciation-dict](https://github.com/DanielSWolf/wiki-pronunciation-dict)
   - [wikipron](https://github.com/CUNY-CL/wikipron)



### Docker image for *espeak-ng*  
For some phonetically regular languages, a rule-based G2P system works quite well. This can be done with *espeak-ng*. However, since the compilation of *espeak-ng* is non-trivial, we have provided a docker image of *espeak-ng* for quick use.  
The Docker image for *espeak-ng* is [available on Docker hub](https://hub.docker.com/r/lukeum/espeak-ng).
You can use *espeak-ng* to perform G2P using the following code. 
```
docker pull lukeum/espeak-ng
```
Please refer to espeak-ng's [user guide](https://github.com/espeak-ng/espeak-ng/blob/master/src/espeak-ng.1.ronn) for a tutorial.

You can also convert it into a singularity container.



### Disclaimer

This tool is a beta version and is still under active development. It may have bugs and quirks, alongside the difficulties and provisos which are described throughout the documentation. 
This tool is distributed under MIT license. Please see [license](https://github.com/lingjzhu/charsiu/blob/main/LICENSE) for details. 

By using this tool, you acknowledge:

* That you understand that this tool does not produce perfect camera-ready data, and that all results should be hand-checked for sanity's sake, or at the very least, noise should be taken into account.

* That you understand that this tool is a work in progress which may contain bugs.  Future versions will be released, and bug fixes (and additions) will not necessarily be advertised.

* That this tool may break with future updates of the various dependencies, and that the authors are not required to repair the package when that happens.

* That you understand that the authors are not required or necessarily available to fix bugs which are encountered (although you're welcome to submit bug reports to Jian Zhu (lingjzhu@umich.edu), if needed), nor to modify the tool to your needs.

* That you will acknowledge the authors of the tool if you use, modify, fork, or re-use the code in your future work.  

* That rather than re-distributing this tool to other researchers, you will instead advise them to download the latest version from the website.

... and, most importantly:

* That neither the authors, our collaborators, nor the the University of Michigan or any related universities on the whole, are responsible for the results obtained from the proper or improper usage of the tool, and that the tool is provided as-is, as a service to our fellow linguists.

All that said, thanks for using our tool, and we hope it works wonderfully for you!

### Contact
Please contact Jian Zhu ([lingjzhu@umich.edu](lingjzhu@umich.edu)) for technical support.  
Contact Cong Zhang ([cong.zhang@ru.nl](cong.zhang@ru.nl)) if you would like to receive more instructions on how to use the package.

