#import all important libraries
import os 
import json 
import numpy as np 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd

#this is the class text_dataset. this inherites Dataset class. this will be used by dataLoader to load tensor data during training
class text_dataset(Dataset):
    #costructor
    def __init__(self, x_y_list, vocab_path, max_seq_length=256, vocab = 'base-cased', transform=None):
        #gives max_seq_lenght
        self.max_seq_length = max_seq_length
        #initializes training data
        self.x_y_list = x_y_list
        #initializes vocab
        self.vocab = vocab
        #initialzes tokenizer according to the vocab used
        if self.vocab == 'base-cased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=True)
        elif self.vocab == 'finance-cased':
            self.tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = False, do_basic_tokenize = True)
        elif self.vocab == 'base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True) 
        elif self.vocab == 'finance-uncased':
            self.tokenizer = BertTokenizer(vocab_file = vocab_path, do_lower_case = True, do_basic_tokenize = True)
    #this is the function which returns the tokenized item. its is important to declare this function for Dataset and DataLoader
    def __getitem__(self,index):
        #this gets the tokenized review we need to return. This just split sentence into words and also does some processing of the text
        tokenized_review = self.tokenizer.tokenize(self.x_y_list[0][index])
#         inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
#         print(inputs)
        #truncating if review length > max_seq_lenght
        if len(tokenized_review) > self.max_seq_length:
            tokenized_review = tokenized_review[:self.max_seq_length]
            #here we are trying to create a token to id from vocab. It replaces every token from its postion in vocab
        ids_review  = self.tokenizer.convert_tokens_to_ids(tokenized_review)
        #Here we are trying to make a mask. This tells which input word the model needs to predict.. Here all are one. This means there is none that the model needs to predict
        mask_input = [1]*len(ids_review)
        #this makes the padding and add to both the lists
        padding = [0] * (self.max_seq_length - len(ids_review))
        ids_review += padding
        mask_input += padding
        
        input_type = [0]*self.max_seq_length  
        
        assert len(ids_review) == self.max_seq_length
        assert len(mask_input) == self.max_seq_length
        assert len(input_type) == self.max_seq_length 
        
        ids_review = torch.tensor(ids_review)
        mask_input =  torch.tensor(mask_input)
        input_type = torch.tensor(input_type)
        #This gets the correct label
        sentiment = self.x_y_list[1][index] 
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        input_feature = {"token_type_ids": input_type, "attention_mask":mask_input, "input_ids":ids_review}
        #returns tensor for label and input_features
        return input_feature, list_of_labels[0]
    
    def __len__(self):
        #returns the lenght of training data
        return len(self.x_y_list[0])


def transform_labels(x_y_list):
    #this is the dict_labels dictonary
    dict_labels = {'positive': 0, 'neutral':1, 'negative':2}
    x_y_list_transformed = [[item[0], dict_labels[item[1]]] for item in x_y_list]
    X = np.asarray([item[0] for item in x_y_list_transformed])
    y = np.asarray([item[1] for item in x_y_list_transformed])
    return X, y

def financialPhraseBankDataset(dir_):
    #this function preprocess the data and splits it into test and train
    fb_path = os.path.join(dir_, 'FinancialPhraseBank-v1.0')
    data_50 = os.path.join(fb_path, 'Sentences_50Agree.txt')
    sent_50 = []
    rand_idx = 45
    
    with open(data_50, 'rb') as fi:
        for l in fi:
            l = l.decode('utf-8', 'replace')
            sent_50.append(l.strip())
    #also preprocess data
    x_y_list_50 = [sent.split("@") for sent in sent_50]
    x50, y50 = transform_labels(x_y_list_50)
    
    data = [x50, y50]
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.1, random_state=rand_idx, stratify=data[1])

    y_train = pd.get_dummies(y_train).values.tolist()
    y_test = pd.get_dummies(y_test).values.tolist()
    X_train = X_train.tolist()
    X_test = X_test.tolist()
            
    final_data = [X_train, X_test, y_train, y_test] 
     
    return final_data