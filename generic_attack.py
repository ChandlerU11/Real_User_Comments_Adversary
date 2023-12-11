import torch
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import pandas as pd
import argparse
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import random
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train Text CNN classificer')
parser.add_argument(
    '-dataset',
    type=str,
    default="gossipcop",
    help='dataset to use')

parser.add_argument(
    '-model',
    type=str,
    default="gossipcop",
    help='model to use for prediction')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 0
else:
    device = 'cpu'
device = 'cpu'
# Load the model
model = SentenceTransformer("johngiorgi/declutr-small", device=device)

df = pd.read_csv('../fake_news_data/'+ args.dataset + '_train.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
#test = pd.read_csv('../fake_news_data/'+ args.dataset + '_test.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
#df = pd.concat([train, test]).reset_index()
df['content'] = [' '.join(each) for each in df['content']]
df['content_embeddings'] = model.encode(df['content'].tolist()).tolist()

comment_df = pd.read_csv('attack_candidate_files/comment_influence_' + args.dataset + '_' + args.model + '.csv')
comment_df['comment_embedding'] = model.encode(comment_df['comments'].tolist()).tolist()

def find_dissimilar_comms(df):
    gen_attacks = []
    for i in tqdm(range(len(df))):
        distance = cdist([df.loc[i]['content_embeddings']], [df.loc[i]['comment_embedding']], metric='cosine')[0]
        if distance >= .8:
            gen_attacks.append(df.loc[i]['comments'])
        #att = list(zip(df['comments'].tolist(), distances))
        #gen_attacks.extend([comment for comment, dist in att if dist >= .8])
    return gen_attacks

pos_df = df[df['label'] == 1]
neg_df = df[df['label'] == 0]

#Need to join pos_df and neg_df with comment_df that meets diff condition
#print(comment_df[comment_df['label'] == 0])
#print(comment_df[comment_df['label'] == 1])
#print(comment_df[comment_df['conf_fake_diff'] < 0])
#print(comment_df[comment_df['conf_fake_diff'] > 0])
#print(comment_df.loc[(comment_df['conf_fake_diff'] < 0) & (comment_df['label'] == 0)])
#print(comment_df.loc[(comment_df['conf_fake_diff'] > 0) & (comment_df['label'] == 1)])
#print(pos_df[['id', 'content']].merge(comment_df.loc[(comment_df['conf_fake_diff'] > 0) & (comment_df['label'] == 1)], how = 'inner', on = 'id')[['id', 'comments', 'label']])
#print(neg_df[['id', 'content']].merge(comment_df.loc[(comment_df['conf_fake_diff'] < 0) & (comment_df['label'] == 0)], how = 'inner', on = 'id')[['id', 'comments', 'label']])

pos_df = pos_df[['id', 'content', 'content_embeddings']].merge(comment_df.loc[(comment_df['conf_fake_diff'] > 0) & (comment_df['label'] == 1)], how = 'inner', on = 'id')
neg_df = neg_df[['id', 'content', 'content_embeddings']].merge(comment_df.loc[(comment_df['conf_fake_diff'] < 0) & (comment_df['label'] == 0)], how = 'inner', on = 'id')

print(pos_df.columns)
generic_comms_towards_fake = find_dissimilar_comms(pos_df)
generic_comms_towards_real = find_dissimilar_comms(neg_df)
print(generic_comms_towards_fake, generic_comms_towards_real)
df_out = pd.DataFrame()
label = [1] * len(generic_comms_towards_fake)
label.extend([0] * len(generic_comms_towards_real))
generic_comms_towards_fake.extend(generic_comms_towards_real)
df_out['comment'] = generic_comms_towards_fake
df_out['label'] = label
print(df_out)
df_out.to_csv('attack_candidate_files/generic_attack_comments_' + args.dataset + '_' + args.model + '.csv', index = False)

