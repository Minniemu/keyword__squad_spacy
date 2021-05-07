#squad
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import json
import re
#Load the data
import joblib
data = pd.read_csv("result.csv", encoding="latin1").fillna(method="ffill")
def tokenize(t):
    t = re.sub("\"","",t)
    t = re.sub("' ","'",t)
    t = re.sub(" '","'",t)
    t = re.sub(" -","-",t)
    t = re.sub("- ","-",t)
    t = re.sub(", ",",",t)
    t = re.sub(" ,",",",t)
    return t
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
filename = 'finalized_model.sav'
model = joblib.load(filename)
model.to(device)
model.eval()
test_sentence = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."""
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()
with torch.no_grad():
    output = model(input_ids)

label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
print("label_indices",label_indices)
# join bpe split tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
print(tokenized_sentence)
print(input_ids)
print(label_indices)
print(tokens)
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)
for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))
'''for i in range(0,len(new_tokens)):
    if new_labels[i] == 'A':
        print(new_tokens[i]+"\t" ,end = '')
        if new_labels[i+1] != 'A' and i+1 < len(new_tokens):
            print("\n") '''
temp_list = []
temp = ""
for i in range(0,len(new_tokens)):
    if new_labels[i] == 'A':
        temp += new_tokens[i]
        temp += " "
        if new_labels[i+1] != 'A' and i+1 < len(new_tokens):
            temp_list.append(temp)
            temp = ""
answer_list = [tokenize(a) for a in temp_list]
result = {}
result['context'] = test_sentence
result['answer_list'] = answer_list
print("answer_list = ",answer_list)
with open('result.json','w') as writefile:
    json.dump(result,writefile)