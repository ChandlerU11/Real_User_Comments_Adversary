import sys
import csv
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
import operator
import sys
import pickle
from ast import literal_eval
import os
from tqdm import tqdm
from scipy.spatial.distance import cosine
import random
import argparse

parser = argparse.ArgumentParser(description='Train Text CNN classificer')
parser.add_argument(
    '-dataset',
    type=str,
    default="gossipcop",
    help='dataset to use')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 0
else:
    device = 'cpu'
device = 'cpu'
# Load the model
model = SentenceTransformer("johngiorgi/declutr-small", device=device)

train = pd.read_csv('~/fake_news_data/'+ args.dataset + '_train.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
test = pd.read_csv('~/fake_news_data/'+ args.dataset + '_test.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})

train['content'] = [' '.join(each) for each in train['content']]
test['content'] = [' '.join(each) for each in test['content']]

embeddings = model.encode(test['content'].tolist())
test['embeddings'] = embeddings.tolist()

embeddings = model.encode(train['content'].tolist())
train['embeddings'] = embeddings.tolist()

def find_comms(df1, df2):
  single_attack_comms = []
  pos_attack_comms = []
  for i in tqdm(range(len(df1))):
    embeds = []
    for x in range(len(df2['embeddings'])):
      embeds.append(cosine(df1.loc[i]['embeddings'], df2.loc[x]['embeddings']))
    zz = list(zip(list(df2['comments']),embeds))
    att = sorted(zz, key = lambda x: x[1])
    att = [x for x in att if len(x[0]) > 0]
    try:
      single_attack_comms.append(random.choice(att[0][0]))
    except:
      print(att[0][0])
    comm_list = []
    for each in att:
      comm_list.extend(each[0])
    pos_attack_comms.append(comm_list[:30])
  return single_attack_comms, pos_attack_comms

test_neg = test[test['label'] == 0].reset_index()
test_pos = test[test['label'] == 1].reset_index()


test_pos['1_attack_comm'], test_pos['30_attack_comm'] = find_comms(test_pos, train[train['label'] == 0].reset_index())
test_neg['1_attack_comm'], test_neg['30_attack_comm'] = find_comms(test_neg, train[train['label'] == 1].reset_index())

pos_out = test_pos[['id', 'label', '1_attack_comm', '30_attack_comm']]
neg_out = test_neg[['id', 'label', '1_attack_comm', '30_attack_comm']]
df_out = pd.concat([pos_out, neg_out])
df_out.to_csv('attack_candidate_files/copycat_attack_' + args.dataset + '.csv')