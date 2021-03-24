import spacy     
import string                                             
class detokenizer:                                                                            
    """ This class is an attempt to detokenize spaCy tokenized sentence """
    def __init__(self, model="en_core_web_sm"):             
        self.nlp = spacy.load(model)
     
    def __call__(self, tokens : list):
        """ Call this method to get list of detokenized words """                     
        while self._connect_next_token_pair(tokens):
            pass              
        return tokens                                                                         
                                               
    def get_sentence(self, tokens : list) -> str:                                                                                                                                            
        """ call this method to get detokenized sentence """            
        return " ".join(self(tokens))
                                               
    def _connect_next_token_pair(self, tokens : list):                  
        i = self._find_first_pair(tokens)
        if i == -1:                                                                                                                                                                          
            return False                                                                                                                 
        tokens[i] = tokens[i] + tokens[i+1]                                                   
        tokens.pop(i+1)                                                                                                                                                                       
        return True                                                                                                                                                                          
                                                                                                                                                                                             
                                                                                                                                                                                             
    def _find_first_pair(self,tokens):                                                                                                                                                       
        if len(tokens) <= 1:                                                                                                                                                                 
            return -1                                                                         
        for i in range(len(tokens)-1):
            if self._would_spaCy_join(tokens,i):                                
                return i
        return -1                                                                             
                                               
    def _would_spaCy_join(self, tokens, index):                                       
        """             
        Check whether the sum of lengths of spaCy tokenized words is equal to the length of joined and then spaCy tokenized words...                                                                  
                        
        In other words, we say we should join only if the join is reversible.          
        eg.:             
            for the text ["The","man","."]
            we would joins "man" with "."
            but wouldn't join "The" with "man."                                               
        """                                    
        left_part = tokens[index]
        right_part = tokens[index+1]
        length_before_join = len(self.nlp(left_part)) + len(self.nlp(right_part))
        length_after_join = len(self.nlp(left_part + right_part))
        if self.nlp(left_part)[-1].text in string.punctuation:
            return False
        return length_before_join == length_after_join 
#squad+spacy
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
#Load the data
import joblib
data = pd.read_csv("result2.csv", encoding="latin1").fillna(method="ffill")
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(data)
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
labels = [[s[2] for s in sentence] for sentence in getter.sentences]
tag_values = list(set(data["Tag"].values))
tag_values.append("PAD")
tag_values.sort()
tag2idx = {t: i for i, t in enumerate(tag_values)}

print(tag_values)
print(tag2idx)
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel
#import transformers
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#print(torch.__version__)

MAX_LEN = 75
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
import joblib
import json
filename = 'finalized_model.sav'
model = joblib.load(filename)
model.to(device)
model.eval()
sen_num = 1
articles = []
result = []
#articles['version'] = 2.0
#articles['data'] = []
with open('article.json') as readfile:
    
    read = json.load(readfile)
    data = read['data']
    for i in range(0,50):
        d = data[i]
        test_sentence = d['article']
        #articles.append(d['article'])
        
        tokenized_sentence = tokenizer.encode(test_sentence)
        input_ids = torch.tensor([tokenized_sentence]).cuda()
        with torch.no_grad():
            output = model(input_ids)

        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        #print("label_indices",label_indices)
        # join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        #print(tokenized_sentence)
        #print(input_ids)
        #print(label_indices)
        #print(tokens)
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
        #for token, label in zip(new_tokens, new_labels):
            #print("{}\t{}".format(label, token))
        keyword = []
        a = {}
        dt = detokenizer()                     

        for i in range(0,len(new_tokens)):
            print("Sentence num = ",sen_num)
            sen_num += 1

            if new_labels[i] == 'A':
                keyword.append(new_tokens[i])
                if i+1 >= len(new_tokens):
                    break
                if new_labels[i+1] != 'A' :
                    temp = {}
                    a['context'] = test_sentence
                    a['answers'] = []
                    detokenized_sentence = dt.get_sentence(keyword)
                    temp['answer_start'] = test_sentence.find(detokenized_sentence)
                    temp['text'] = detokenized_sentence
                    a['answers'].append(temp)
                    keyword = []
                    a = {}
                    result.append(a)
                    print("\n") 
#print("result",result)
with open('qg_data.json','w') as writefile:
    json.dump(result,writefile)