# Real_User_Comments_Adversary
Implementation of Copycat, Generic, and Topic-Specific attacks for fake news detectors. For necessary repo locations, see bottom of README.

## Workflow
1. Clean fake news data (skip if already done  for ReST)
2. Train classifier (dEFEND, RoBERTa, or TextCNN) on fake news dataset (skip if already done  for ReST)
3. Start desired classifier prediction script
4. Run generic, specific, and copycat attack scripts
5. Repeat step 3-4 for all fake news classifiers

## Create Conda Env "real_comms"
`conda env create -f env.yml`

## Find Individual Comment Influence on Classification
`python find_comment_influence.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend]`

## Get CopyCat Comments
`python copycat_attack.py -dataset [gossipcop, politifact]`

*Copycat works independently of attacked model

## Get Generic Comments
`python generic_attack.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend]`

## Get Specific Comments
`python specific_attack.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend]`

## Test
`python test.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend] -attack [copycat, generic, specific, all] -target_label [fake, real] -user_comms [yes, no]`

## Test Copycat with Single Chance
`python test.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend] -attack copycat -target_label [fake, real] -user_comms [yes, no] -copycat_sing yes`

This command produces the results used for CopyCat in the ReST Adversary paper.


## Repo Locations
> Each of these repos contain an env.yml file that allows for dependency installation using anaconda. When running code in these repos remember that you must activate their respective conda environment. Some of these repos use same conda enviroment as the ReST_Adversary which is noted in their README files. For adversarial training (workflow step 5) to work properly, this repo and all the other repos need to be cloned into the same root folder and to remain in separate folders.

Data with cleaning script - https://github.com/ChandlerU11/fake_news_data

ReST - https://github.com/ChandlerU11/ReST_Adversary

dEFEND - https://github.com/ChandlerU11/dEFEND_Surrogate

RoBERTa - https://github.com/ChandlerU11/RoBERTa_Surrogate

TextCNN - https://github.com/ChandlerU11/TextCNN_Surrogate
