# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:56:35 2021

@author: rania
"""
import string
from nltk.corpus import stopwords
import pickle
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import  defaultdict

class DataFeatures:
    
    def __init__(self, dataset, train_data=None):
        self.dataset = dataset
        self.extra = None
        #load the data
        self.raw = load_dataset()
        #self.raw= prepa(self)
        # preprossesing data 
        prepa_data(self)
        #split data to test/train data
        self.train, self.validation, self.test=split_data(self)
        self.count_matrix, self.tfidf_matrix = None, None
        self.count_matrix = numberofwords(self)   
        self.tfidf_matrix = tfidf(self)
        self.labels_train = self.train.MOS.values
        self.nl_matrix = nlfeatures(self)
        save(self.count_matrix,'wordcount.pkl')
        save(self.labels_train,'label.pkl')
        save(self.nl_matrix,'nl.pkl')
        
def load_dataset():
        return pd.read_excel('C:/Users/rania/Desktop/training.xlsx')
    
def prepa(self):
    df = self.raw
    df['split'] = np.random.choice(3, len(df), p=[0.8, 0.1,0.1])
    df.to_excel('C:/Users/rania/Desktop/training.xlsx', index=False)
    return df

# Define the function to remove the punctuation
def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))
# Apply to the DF series
  
def prepa_data(self):
    self.raw.Sentence = self.raw.Sentence.replace({'ä':'ae','ü':'ue','ö':'oe'},regex=True)
    #self.raw.MOS= self.raw.MOS.apply(int)
    df = self.raw
    df['Sentence'] = df['Sentence'].apply(remove_punctuations)  
    return df

def split_data(self):
        df = self.raw
        train = df[df.split == 0]
        validation = df[df.split==1]
        test = df[df.split==2]
        return train, validation, test
    
def numberofwords(self):
     train = self.train
     german_stop_words = stopwords.words('german')
     vectorizer = CountVectorizer(ngram_range=(1,3) ,stop_words = german_stop_words)
     vector= vectorizer.fit(train.Sentence)
     #     vector = vectorizer.transform(vector)
     vector = vectorizer.transform(train.Sentence)
     self.count_matrix= vector.toarray()
     return self.count_matrix
 
def tfidf(self, tfidf_params={}):
        # Operate on training only for fitting
        train = self.train
        tfidf = TfidfVectorizer(**tfidf_params)
        tfidf.fit(train.Sentence)
        self.tfidf = tfidf
        self.tfidf_matrix = tfidf.transform(train.Sentence).todense()
        return self.tfidf_matrix  
    
def nlfeatures(self):

        # Function to check if the token is a noise or not
        def is_noise(token, noisy_pos_tags=['PROP'], min_token_length=2):
            return token.pos_ in noisy_pos_tags or token.is_stop or len(token.string) <= min_token_length

        def cleanup(token):
            return token.lower().strip()
        
        nlp = spacy.load("en_core_web_sm")
        feature_matrix = []
        num_docs = 0
        for text in self.train.Sentence:
            doc = nlp(text)

            num_docs += 1
            POS_TAGS = [
                "", "ADJ", "ADP", "ADV", "AUX", "CONJ",
                "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
                "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB",
                "X", "EOL", "SPACE"
            ]
            noun_chunks = list(doc.noun_chunks)
            sentences = list(doc.sents)
            avg_sent_length = sum([len(sent) for sent in sentences]) / len(sentences)
            all_tag_counts = defaultdict(int)
            for w in doc:
                all_tag_counts[w.pos_] += 1
                if w.pos_ == 'PUNCT':
                    all_tag_counts[w] += 1
            # cleaned_list = [cleanup(word.string) for word in doc if not is_noise(word)]
            feats = []
            for tag in POS_TAGS:
                feats.append(all_tag_counts[tag])
            feats.append(len(noun_chunks))          # num_noun_chunks
            feats.append(len(sentences))            # num_sentences
            feats.append(avg_sent_length)
            feature_matrix.append(feats)
        self.nl_matrix = np.array(feature_matrix)  
        return self.nl_matrix   
    
def save(self,text):
    pickle.dump(self,open(text,'wb'))
    
if __name__ == "__main__":
    x = DataFeatures('dataset')