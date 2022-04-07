# CharsiuG2P

### Introduction
CharsiuG2P is transformer based tool for grapheme-to-phoneme conversion in over 100 languages. Given an orthographic word, CharsiuG2P predicts its pronunciation through a neural G2P model.

This repository also contains instructions to replicate the papaer *ByT5 model for massively multilingual grapheme-to-phoneme conversion*.  

### Table of content
- [Introduction](https://github.com/lingjzhu/CharsiuG2P#introduction)
- [Usage](https://github.com/lingjzhu/CharsiuG2P#usage)
- [Pretrained models](https://github.com/lingjzhu/CharsiuG2P/blob/main/README.md#pretrained-models)
- [Training and fine-tuning](https://github.com/lingjzhu/CharsiuG2P#training-and-fine-tuning)
- [Docker image for *espeak-ng* ](https://github.com/lingjzhu/CharsiuG2P#docker-image-for-espeak-ng)
- [G2P Datasets](https://github.com/lingjzhu/CharsiuG2P#g2p-datasets)
- [Disclaimer](https://github.com/lingjzhu/CharsiuG2P/blob/main/README.md#disclaimer)
- [Contact](https://github.com/lingjzhu/CharsiuG2P/blob/main/README.md#contact)


### Usage

### Pretrained models
Pretrained models are hosted at [HuggingFace model hub](https://huggingface.co/charsiu) with the prefix "G2P". 

### Training and fine-tuning

### Docker image for *espeak-ng*  
For some phonetically regular languages, a rule-based G2P system works quite well. This can be done with *espeak-ng*. However, since the compilation of *espeak-ng* is non-trivial, we have provided a docker image of *espeak-ng* for quick use.  
The Docker image for *espeak-ng* is [available on Docker hub](https://hub.docker.com/r/lukeum/espeak-ng).
You can use *espeak-ng* to perform G2P using the following code. 
```
docker pull lukeum/espeak-ng
```
Please refer to espeak-ng's [user guide](https://github.com/espeak-ng/espeak-ng/blob/master/src/espeak-ng.1.ronn) for a tutorial.

You can also convert it into a singualrity container.


### G2P Datasets
A detailed catalogue of pronunciation dictionaries with downloadable links can be found in [this form](https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?usp=sharing).  

- Word lists are acquired via [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download).

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

