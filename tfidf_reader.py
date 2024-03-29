# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.arlstem import ARLSTem
from  nltk.stem import PorterStemmer,LancasterStemmer
from nltk.tokenize import WordPunctTokenizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

class TfidfReader:
    SYMBOLS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"'
    def __init__(self, P):
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = PorterStemmer()
        self.docs = self.get_answer_canditates(P)
        docs_stem = []
        for doc in self.docs:
            docs_stem.append(self.stem_string(doc))
        # self.stopwords = stopwords.words('arabic')
        # self.stopwords = open("stopwords-ur.txt").read().splitlines()
        # self.stopwords=[i.lower() for i in self.stopwords]
        self.stopwords=set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 4), norm=None)#, stop_words=self.stopwords)
        self.tfidf_matrix = self.vectorizer.fit_transform(docs_stem)


    def stem_string(self, str):
        str_tokens = self.tokenizer.tokenize(str)
        str_processed = ""
        for token in str_tokens:
            has_symbol = False
            for s in self.SYMBOLS:
                if s in token:
                    has_symbol = True
                    break
            if not has_symbol:
                str_processed += token +  " " + self.stemmer.stem(token) +" "
        return str_processed

    def concatenateString(self, paragraph, start, length):
        final_string = paragraph[start]
        for i in range(1, length):
            final_string += " " + paragraph[start + i]
        return final_string

    def get_answer_canditates(self, paragraph):
        para_sents = nltk.sent_tokenize(paragraph)
        candidates = []
        for sent in para_sents:
            para_words = sent.split()
            for i in range(0, len(para_words)):
                for j in range(1, min(15, len(para_words) - i + 1)):
                    candidate = self.concatenateString(para_words, i, j)
                    candidates.append(candidate)
        return candidates


    def read(self, P , Q):
        Q = self.stem_string(Q)
        query_tfidf = self.vectorizer.transform([Q])
        similarities_raw = cosine_similarity(self.tfidf_matrix, query_tfidf)
        similarities = []
        for s in similarities_raw:
            similarities.append(s[0])
        max_index = np.argmax(similarities)

        return self.docs[max_index]

def Test_TfidfReader():
    P = "میرا نام ہاسین ہے اور میرے والد یونیورسٹی میں کام کرتے ہیں۔ میری والدہ ایک قاتل تھیں ، اس نے میرے والد کو مار ڈالا"
    reader = TfidfReader(P)
    print(reader.read("میرا نام ہاسین ہے اور میرے والد یونیورسٹی میں کام کرتے ہیں۔ میری والدہ ایک قاتل تھیں ، اس نے میرے والد کو مار ڈالا",
        "یونیورسٹی میں. میری والدہ ایک قاتل تھیں"))



