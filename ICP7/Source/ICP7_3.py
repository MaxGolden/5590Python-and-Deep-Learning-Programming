"""
3. Apply the following on the text and show output:
Tokenization
Stemming
POS
Lemmatization
Trigram
Named Entity Recognition

--- see ntlk.py
"""
import nltk

sentence = open('input.txt', encoding="utf8").read()

# Tokenization
stokens = nltk.sent_tokenize(sentence)
wtokens = nltk.word_tokenize(sentence)

print("\n============== Tokenization ==============\n")

n = 0
for s in stokens:
     n = n + 1
     if n < 4:
         print("+++stoken+++:", s)

n1 = 0
for t in wtokens:
    n1 = n1 + 1
    if n1 < 4:
        print("___wtokens___:", t)

print("\n============== Stremming ==============\n")
# Stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer

pStemmer = PorterStemmer()
lStemmer = LancasterStemmer()
sStemmer = SnowballStemmer('english')

n1 = 0
for t in wtokens:
    n1 = n1 + 1
    if n1 < 4:
        print(pStemmer.stem(t), lStemmer.stem(t), sStemmer.stem(t))

print("\n============== POS / Lemmatization ==============\n")

# POS
# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n1 = 0
for t in wtokens:
    n1 = n1 + 1
    if n1 < 6:
        print("Lemmatizer:", lemmatizer.lemmatize(t), ",    With POS=a:", lemmatizer.lemmatize(t, pos="a"))

print("\n============== Trigram ==============\n")
# Trigram
from nltk.util import ngrams
token = nltk.word_tokenize(sentence)

n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        token = nltk.word_tokenize(s)
        bigrams = list(ngrams(token, 2))
        trigrams = list(ngrams(token, 3))
        print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)

print("\n============== Named Entity Recognition ==============\n")
# Named Entity Recognition
from nltk import word_tokenize, pos_tag, ne_chunk
n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        print(ne_chunk(pos_tag(word_tokenize(s))))
