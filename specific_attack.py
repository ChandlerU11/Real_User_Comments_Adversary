from ast import literal_eval
import pandas as pd
import argparse
from scipy.spatial.distance import cdist
from tqdm import tqdm
import random
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words= nltk.corpus.stopwords.words('english')


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

def clean_text(content):
  cleaned_text = ""
    
  if isinstance(content, str):
    le = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(content)
    tokens = [ le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w) > 2 ]
    cleaned_text = " ".join(tokens)

  return cleaned_text

#train = pd.read_csv('../fake_news_data/'+ args.dataset + '_train.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
df = pd.read_csv('../fake_news_data/'+ args.dataset + '_test.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
#df = pd.concat([train, test]).reset_index()

df['content'] = [' '.join(each) for each in df['content']]
df['clean_content'] = df['content'].apply(clean_text)

comment_df = pd.read_csv('attack_candidate_files/comment_influence_' + args.dataset + '_' + args.model + '.csv')
comment_df['clean_comments'] = comment_df['comments'].apply(clean_text)

vect = TfidfVectorizer(stop_words=stop_words, max_features=1000)


for comp in tqdm(range(3, 21)):
    lda_model = LatentDirichletAllocation(n_components=comp,
                                            learning_method='online', # which algorithm is this?
                                            random_state=42,
                                            max_iter=10)

    vect_text = vect.fit_transform(df['clean_content'])
    compute_lda = lda_model.fit(vect_text)
    df['content_lda'] = compute_lda.transform(vect_text).tolist()

    vect_text = vect.transform(comment_df['clean_comments'])
    comment_df['comment_lda'] = compute_lda.transform(vect_text).tolist()

    pos_df = df[df['label'] == 1].reset_index()
    neg_df = df[df['label'] == 0].reset_index()

    NUM_NEIGHBORS = 30
    neigh = NearestNeighbors(n_neighbors=NUM_NEIGHBORS)
    neg_comms_filt = comment_df.loc[(comment_df['conf_fake_diff'] < 0) & (comment_df['label'] == 0)].reset_index()

    neigh.fit(neg_comms_filt['comment_lda'].tolist())
    attack_comms_spec = []
    for i in range(len(pos_df)):
        att = []
        for each in neigh.kneighbors([pos_df.loc[i]['content_lda']], return_distance=False)[0]:
            att.append(neg_comms_filt.loc[each]['comments'])
        attack_comms_spec.append(att)

    pos_df['attack_comms_spec'] = attack_comms_spec


    #################################################################################################
    neigh = NearestNeighbors(n_neighbors=NUM_NEIGHBORS)
    pos_comms_filt = comment_df.loc[(comment_df['conf_fake_diff'] > 0) & (comment_df['label'] == 1)].reset_index()
    vect_text = vect.transform(pos_comms_filt['clean_comments'])
    pos_comms_filt['comment_lda'] = compute_lda.transform(vect_text).tolist()

    neigh.fit(pos_comms_filt['comment_lda'].tolist())
    attack_comms_spec = []
    for i in range(len(neg_df)):
        att = []
        for each in neigh.kneighbors([neg_df.loc[i]['content_lda']], return_distance=False)[0]:
            att.append(pos_comms_filt.loc[each]['comments'])
        attack_comms_spec.append(att)

    neg_df['attack_comms_spec'] = attack_comms_spec

    df_out = pd.concat([pos_df, neg_df])[['id', 'label', 'attack_comms_spec']]

    df_out.to_csv('attack_candidate_files/specific_attack_comments_lda_'+ str(comp) + '_' + args.dataset + '_' + args.model + '.csv', index = False)
