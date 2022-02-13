
import re
from phonemizer.backend import EspeakBackend

backend = EspeakBackend('af')


words = []
with open('dicts/wordlist/afr','r') as f:
    for line in f.readlines():
	    words.append(line.strip())


lexicon = {word:backend.phonemize([word]) for word in words} 

with open('dicts/wordlist/afr.tsv','w') as out:
    for w,p in lexicon.items():
	    out.write("%s\t%s\n"%(w,p[0]))