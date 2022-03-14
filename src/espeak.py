
import os
import argparse
from phonemizer.backend import EspeakBackend


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--lang',type=str)
parser.add_argument('--outpath', type=str)   
args = parser.parse_args()

words = []
with open(args.path,'r') as f:
	for line in f.readlines():
		words.append(line.strip())


backend = EspeakBackend(language=args.lang,with_stress=True,tie=True)

phones = backend.phonemize(words,strip=True)

assert len(words)==len(phones)


with open(args.outpath,'w') as out:
	
	for w,p in zip(words,phones):
		out.write("%s\t%s\n"%(w,p))

