# Real_User_Comments_Adversary
Implementation of Copycat, Generic, and Topic-Specific attacks for fake news detectors. Uses the fake_news_data repo (https://github.com/ChandlerU11/fake_news_data) and interacts with surrogate classifiers in same manner as described in the ReST adversary README (https://github.com/ChandlerU11/ReST_Adversary). 

## Create Conda Env "real_comms"
`conda env create -f env.yml`

## Get CopyCat Comments
`python copycat_attack.py -dataset [gossipcop, politifact]`

*Copycat works independently of attacked model

## Get Generic Comments
`python generic_attack.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend]`

## Get Specific Comments
`python specific_attack.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend]`

## Test Attack Effectiveness
`python test.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend] -attack [copycat, generic, specific, all] -target_label [fake, real] -user_comms [yes, no]`
