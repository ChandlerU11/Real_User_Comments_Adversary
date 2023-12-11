import argparse
import pandas as pd 
import socket
from ast import literal_eval
import random
import time
import os
import pickle

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

parser.add_argument(
    '-attack',
    type=str,
    default="all",
    help='model to use for prediction')


parser.add_argument(
    '-target_label',
    type=str,
    default='real',
    help='model to use for prediction')


parser.add_argument(
    '-user_comms',
    type=str,
    default=False,
    help='model to use for prediction')

args = parser.parse_args()

def wait(originalTime, fileName):
    while(os.path.getmtime(fileName) <= originalTime):
        time.sleep(1)
    time.sleep(3)

def get_preds(test_sing = False):
    fileName = '../ReST_Temp_Files/'+ args.model +'_preds_post'
    originalTime = os.path.getmtime(fileName)
    
    print("Calling " + args.model)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 9988))
    if test_sing == False:
        s.sendall(b'Test Multiple')
    else:
        s.sendall(b'Test Single')

    s.close()
    
    wait(originalTime, fileName)

    infile = open('../ReST_Temp_Files/'+ args.model +'_preds_pre','rb')
    pre = pickle.load(infile, encoding = 'latin1')
    infile.close()

    fileName = '../ReST_Temp_Files/'+ args.model +'_preds_post'
    
    infile = open(fileName,'rb')
    post = pickle.load(infile, encoding = 'latin1')
    infile.close()
    print("Test Run")
    return pre, post

def copycat_attack(df):
    cand_df = pd.read_csv('attack_candidate_files/copycat_attack' + args.dataset + '.csv', converters = {'30_attack_comm':literal_eval})
    df = df.merge(cand_df[['id','30_attack_comm']], how = 'inner', on = 'id')
    if args.target_label == 'real':
        df = df[df['label'] == 1].reset_index()
    else:
        df = df[df['label'] == 0].reset_index()
    
    out_df = []
    for i in range(len(df)):
       test_df = pd.concat([df.loc[i].drop(columns = ['30_attack_comm']).to_frame().transpose()] * 30)
       test_df['response'] = df.loc[i]['30_attack_comm']
       out_df.append(test_df)
    out_df = pd.concat(out_df)
    
    t5_gen_file = '../ReST_Temp_Files/T5_training_step_gen.csv'
    ot = os.path.getmtime(t5_gen_file)
    out_df.to_csv(t5_gen_file, escapechar = '\\')
    wait(ot,t5_gen_file)
    
    print(args.user_comms)
    pre, post = get_preds(test_sing = args.user_comms)
    print(pre,post)
    out_df['pre'] = pre
    out_df['post'] = post
    if args.target_label == 'real':
        print(out_df.loc[(out_df['label'] == 1) & (out_df['pre'] == 1) & (out_df['post'] == 0)])
        eff = len(list(set(out_df.loc[(out_df['label'] == 1) & (out_df['pre'] == 1) & (out_df['post'] == 0)]['id'].tolist()))) / len(list(set(out_df.loc[(out_df['label'] == 1) & (out_df['pre'] == 1)]['id'].tolist()))) 
        print(eff)
    else:
        print(out_df.loc[(out_df['label'] == 0) &(out_df['pre'] == 0) & (out_df['post'] == 1)])
        eff = len(list(set(out_df.loc[(out_df['label'] == 0) & (out_df['pre'] == 0) & (out_df['post'] == 1)]['id'].tolist()))) / len(list(set(out_df.loc[(out_df['label'] == 0) & (out_df['pre'] == 0)]['id'].tolist()))) 
        print(eff)
    return eff

def specific_attack(df):
    return eff

def generic_attack(df):
    cand_df = pd.read_csv('attack_candidate_files/generic_attack_comments_' + args.dataset + '_' + args.model + '.csv')
    if args.target_label == 'real':
        df = df[df['label'] == 1].reset_index()
        cand_list = cand_df[cand_df['label'] == 0]['comment'].tolist()
    else:
        df = df[df['label'] == 0].reset_index()
        cand_list = cand_df[cand_df['label'] == 1]['comment'].tolist()
    
    out_df = []
    for i in range(len(df)):
       test_df = pd.concat([df.loc[i].to_frame().transpose()] * 30)
       test_df['response'] = random.sample(cand_list, 30)
       out_df.append(test_df)

    out_df = pd.concat(out_df)
    print(out_df)

    t5_gen_file = '../ReST_Temp_Files/T5_training_step_gen.csv'
    ot = os.path.getmtime(t5_gen_file)
    out_df.to_csv(t5_gen_file, escapechar = '\\')
    wait(ot,t5_gen_file)
    
    print(args.user_comms)
    pre, post = get_preds(test_sing = args.user_comms)
    print(pre,post)
    out_df['pre'] = pre
    out_df['post'] = post
    if args.target_label == 'real':
        print(out_df.loc[(out_df['label'] == 1) & (out_df['pre'] == 1) & (out_df['post'] == 0)])
        eff = len(list(set(out_df.loc[(out_df['label'] == 1) & (out_df['pre'] == 1) & (out_df['post'] == 0)]['id'].tolist()))) / len(list(set(out_df.loc[(out_df['label'] == 1) & (out_df['pre'] == 1)]['id'].tolist()))) 
        print(eff)
    else:
        print(out_df.loc[(out_df['label'] == 0) &(out_df['pre'] == 0) & (out_df['post'] == 1)])
        eff = len(list(set(out_df.loc[(out_df['label'] == 0) & (out_df['pre'] == 0) & (out_df['post'] == 1)]['id'].tolist()))) / len(list(set(out_df.loc[(out_df['label'] == 0) & (out_df['pre'] == 0)]['id'].tolist()))) 
        print(eff)
    return eff

test = pd.read_csv('~/fake_news_data/'+ args.dataset + '_test.csv', converters = {'title':literal_eval,'content':literal_eval,'comments':literal_eval})
eff = copycat_attack(test)
#eff = generic_attack(test)