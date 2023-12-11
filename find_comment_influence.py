import socket
import pandas as pd
import argparse
import time
import os
from ast import literal_eval
from tqdm import tqdm
import pickle 

def wait(originalTime, fileName):
    while(os.path.getmtime(fileName) <= originalTime):
        time.sleep(1)
    time.sleep(3)

def get_preds():
    fileName = '../ReST_Temp_Files/'+ args.model +'_preds_towards_fake'
    originalTime = os.path.getmtime(fileName)

    print("Calling " + args.model)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 9988))
    s.sendall(b'Comment Influence')
    s.close()

    wait(originalTime, fileName)

    infile = open(fileName ,'rb')
    rewards = pickle.load(infile, encoding = 'latin1')
    infile.close()
    return(rewards)

def get_single_comm(df):    
    ids, titles, comms, resp, labels, cont, conf_fake = [], [], [], [], [], [], []
        
    for i in tqdm(range(len(df))):
        for each in df.loc[i]['comments']:
            ids.append(df.loc[i]['id'])
            labels.append(df.loc[i]['label'])
            titles.append(df.loc[i]['title'])
            comms.append([each])
            cont.append(df.loc[i]['content'])
            conf_fake.append(df.loc[i]['conf_fake_pre'])

    df2 = pd.DataFrame()
    df2['id'] = ids
    df2['comments'] = comms
    df2['label'] = labels
    df2['content'] = cont
    df2['title'] = titles
    df2['conf_fake_pre'] = conf_fake
    
    df2 = df2[df2['comments'] != "\n"]
    df2 = df2.dropna()
    return df2

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

df = pd.read_csv('~/fake_news_data/'+ args.dataset + '_train.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
#test = pd.read_csv('~/fake_news_data/'+ args.dataset + '_test.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
#df = pd.concat([train, test]).reset_index()
df_no_comm = df.copy()
df_no_comm['comments'] = [[] for x in df_no_comm['comments']]

print(df_no_comm[['content', 'comments']])
t5_gen_file = '../ReST_Temp_Files/T5_training_step_gen.csv'
ot = os.path.getmtime(t5_gen_file)
df_no_comm.to_csv(t5_gen_file, escapechar = '\\')
wait(ot,t5_gen_file)
df['conf_fake_pre'] = get_preds()

df_single = get_single_comm(df)
print(df_single[['comments']])

t5_gen_file = '../ReST_Temp_Files/T5_training_step_gen.csv'
ot = os.path.getmtime(t5_gen_file)
df_single.to_csv(t5_gen_file, escapechar = '\\')
wait(ot,t5_gen_file)

df_single['conf_fake_post'] = get_preds()
df_single['conf_fake_diff'] = df_single['conf_fake_pre'] - df_single['conf_fake_post']
df_single['comments'] = [x[0] for x in df_single['comments']]
df_single = df_single[['id', 'comments', 'label', 'conf_fake_diff']]
df_single.to_csv('comment_influence_' + args.dataset + '_' + args.model + '.csv', index = False)