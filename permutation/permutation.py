#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from uon.preprocessing import preprocessing
import os

# command line arguments
parser = argparse.ArgumentParser(
    description='Permutation tests for testing modes robustness.')
parser.add_argument("-b", "--brain", help="Path to csv file with brain data.")
parser.add_argument("-c", "--cognition", help="Path to csv file with cognition data.")
parser.add_argument("-i", "--info", help="Path to csv file with other information.", default="")
parser.add_argument("-r", "--regress_out_confounds", help="Regress out confounds.", action="store_true")
parser.add_argument("-f", "--confounds", help="List of confounds columns names in the information csv file (e.g. -f 'Age' 'Sex').", nargs='*', default=[])
parser.add_argument("-m", "--mean", help="Average brain measurement (columns 'xxx_left' and 'xxx_right').", action="store_true")
parser.add_argument("-d", "--drop_rate", help="Maximum percentage of missing values without removing the subject/feature.", action="store_true")
parser.add_argument("-d", "--drop_rate", help="Maximum percentage of missing values without removing the subject/feature.", action="store_true")


args = parser.parse_args()

# load data
if os.path.exists(args.brain):
    df_brain = pd.read_csv(args.brain, index_col=0)
else:
    print('Brain csv file does not exists.')
    exit(1)
if os.path.exists(args.cognition):
    df_cogn = pd.read_csv(args.cognition, index_col=0)
else:
    print('Cognition csv file does not exists.')
    exit(1)
if os.path.exists(args.info):
    df_info = pd.read_csv(args.info, index_col=0)
else:
    df_info = pd.DataFrame() # need to add subject name to make it work

# preprocessing
df_brain, df_cogn, df_info, Xbrain, Ycogn = preprocessing(
    df_brain, df_cogn, df_info, args.regress_out_confounds, args.confounds,
    train_hc_only, drop_rate=args.drop_rate, verbose=True)

# permutation tests
pipeline = pls.PLSPipeline(PLSCanonical(n_components=2),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
pls.permutation_test(Xsulci, Ycogn, pipeline)

# plot results

# save results in csv file

exit(0)