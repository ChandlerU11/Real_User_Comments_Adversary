# Real_User_Comments_Adversary

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
`python test.py -dataset [gossipcop, politifact] -model [textcnn, roberta, defend] -attack [copycat, generic, specific, all] -target_label [fake, real] -user_comms [True, False]`