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

df = pd.concat([train,test]).reset_index()
df['content'] = [' '.join(each) for each in df['content']]

pos = df[df['label'] == 1]
pos = pos.reset_index()
neg = df[df['label'] == 0]
neg = neg.reset_index()

embeddings = model.encode(pos['content'].tolist())
pos['embeddings'] = embeddings.tolist()

embeddings = model.encode(neg['content'].tolist())
neg['embeddings'] = embeddings.tolist()

print(pos, neg)

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

pos['1_attack_comm'], pos['30_attack_comm'] = find_comms(pos, neg)
neg['1_attack_comm'], neg['30_attack_comm'] = find_comms(neg, pos)

pos_out = pos[['id', 'label', '1_attack_comm', '30_attack_comm']]
neg_out = neg[['id', 'label', '1_attack_comm', '30_attack_comm']]
df_out = pd.concat([pos_out, neg_out])
df_out.to_csv('copycat_attack' + args.dataset + '.csv')