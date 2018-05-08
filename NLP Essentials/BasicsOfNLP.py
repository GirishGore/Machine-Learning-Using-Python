# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:39:45 2018

@author: Girish
"""

## BeutifulSoup for extracting information directly from URL
from bs4 import BeautifulSoup 
import urllib.request

## The core NLP library in python
## Other standard libraries with support in java are GATE NLP and Stanford NLP
import nltk
### nltk.download()
 
response = urllib.request.urlopen('http://php.net/')
html = response.read()
soup = BeautifulSoup(html,"html.parser")
text = soup.get_text(strip=True)
### Text has been extracted in the text variables 
print (text)

### Tokenization : Creating tokens out of text
tokens = [t for t in text.split()]

from nltk.text import Text 
### There are few other things we can do with plain text
newText = (Text(tokens))
print(newText)
newText.concordance("PHP")

### Stop Word Removal is the next step. We take help from stopwords cropus.
from nltk.corpus import stopwords
stopwords.words('english')
clean_tokens = tokens[:]
sr = stopwords.words('english')
 
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

## Tokens post removing stop words have been generated here
print (clean_tokens)




##
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet
syn = wordnet.synsets("pain")
print(syn[0].definition())
print(syn[0].examples())


 
print(stemmer.stem('increases'))
freq = nltk.FreqDist(clean_tokens)
for key,val in freq.items():
    print (str(key) + "FrequencyCount : " + str(val) , " Stemming : " , stemmer.stem(key) , " Lemmetization : " , lemmatizer.lemmatize(key))
    syn = wordnet.synsets("pain")
    if(syn.synonyms()):
        print ("Synonym : " + str(syn.synonyms()[0].name() ))
        
freq.plot(20, cumulative=False)

## Lemmatizing Words Using WordNet
## Word lemmatizing is similar to stemming, but the difference is the result of lemmatizing is a real word.
## Unlike stemming, when you try to stem some words, it will result in something like this:

