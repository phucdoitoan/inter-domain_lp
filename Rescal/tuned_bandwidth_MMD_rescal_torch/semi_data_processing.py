

import pandas as pd
#from data_structures import KnowledgeGraph
import torch
import random
from time import time
import os


df_names = ['head', 'rel', 'tail']

#dataset_name = 'FB15k-237' #0.05
dataset_name = 'WN18RR'  #0.2
print('dataset name: ', dataset_name)


train_df = pd.read_csv('../data/%s/train.txt' %dataset_name, delimiter='\t', header=None, names=df_names)
valid_df = pd.read_csv('../data/%s/valid.txt' %dataset_name, delimiter='\t', header=None, names=df_names)
test_df = pd.read_csv('../data/%s/test.txt' %dataset_name, delimiter='\t', header=None, names=df_names)

df = pd.concat([train_df, valid_df, test_df], axis=0)

raw_ent = list(set(df['head'].values).union(set(df['tail'].values)))
print('raw_ent: ', len(raw_ent))

random.shuffle(raw_ent)

ent_size = 7000 #3000
overlap_rate = 0.12 #0.09 #0.15
overlap_size = int(ent_size * overlap_rate)


print('overlap rate: ', overlap_rate)


raw_ent1 = raw_ent[:ent_size]
raw_ent2 = raw_ent[ent_size + overlap_size: 2*ent_size + overlap_size]
raw_ent_common = raw_ent[ent_size: ent_size + overlap_size]

print('raw_ent1: ', len(raw_ent1))
print('raw_ent2: ', len(raw_ent2))
print('raw_ent_common: ', len(raw_ent_common))

head_raw = df['head'].values.tolist()
rel_raw = df['rel'].values.tolist()
tail_raw = df['tail'].values.tolist()

print('head_raw: ', len(head_raw))
print('rel_raw: ', len(rel_raw))
print('tail_raw: ', len(tail_raw))



def raw_ent_filter(df, raw_ent):

    data = []

    for h, r, t in zip(head_raw, rel_raw, tail_raw):
        if (h in raw_ent) and (t in raw_ent):
            data.append([h, r, t])

    return pd.DataFrame(data, columns=df_names)

t0 = time()
raw_df1 = raw_ent_filter(df, raw_ent1)
print('done 1: %.4f s' %(time() - t0), 'raw_df1: ', raw_df1.shape)
raw_df2 = raw_ent_filter(df, raw_ent2)
print('done 2: %.4f s' %(time() - t0), 'raw_df2: ', raw_df2.shape)
raw_df_common = raw_ent_filter(df, raw_ent_common)
print('done _common: %.4f s' %(time() - t0))

print('raw_df1: ', raw_df1.shape)
print('raw_df2: ', raw_df2.shape)
print('raw_df_common: ', raw_df_common.shape)

print('REL number df1: ', len(set(raw_df1['rel'].values)))
print('REL number df2: ', len(set(raw_df2['rel'].values)))
print('REL number df_common: ', len(set(raw_df_common['rel'].values)))

print('Adjusting so that all data share a same set of relations')

common_raw_rel = set(raw_df1['rel'].values).intersection(set(raw_df2['rel'].values))
print('common_raw_rel: ', len(common_raw_rel))

separate_raw_rel = set(df['rel'].values).difference(common_raw_rel)
print('separate_raw_rel: ', len(separate_raw_rel))


def unify_rel(df, separate_raw_rel):

    for rel in separate_raw_rel:
        df = df.drop(df[df['rel'] == rel].index)

    return df

t0 = time()
raw_df1 = unify_rel(raw_df1, separate_raw_rel)
print('done unify_rel 1: %.4f s' %(time() - t0))
raw_df2 = unify_rel(raw_df2, separate_raw_rel)
print('done unify_rel 2: %.4f s' %(time() - t0))
raw_df_common = unify_rel(raw_df_common, separate_raw_rel)
print('done unify_rel _common: %.4f s' %(time() - t0))


print('REL number df1: ', len(set(raw_df1['rel'].values)))
print('REL number df2: ', len(set(raw_df2['rel'].values)))
print('REL number df_common: ', len(set(raw_df_common['rel'].values)))

print('Relation sets of df1, df2 is the same: ', set(raw_df1['rel'].values) == set(raw_df2['rel'].values))
print('Relation set of df_common is in df1: ', set(raw_df_common['rel'].values).issubset(set(raw_df1['rel'].values)))

print('Saving all dataframes into csv files ...')

if not os.path.exists('../data/%s/semi_1.5/divided/' %dataset_name):
    os.makedirs('../data/%s/semi_1.5/divided/' %dataset_name)

raw_df1.to_csv('../data/%s/semi_1.5/divided/train1.csv' %dataset_name, sep='\t', index=False)
raw_df2.to_csv('../data/%s/semi_1.5/divided/train2.csv' %dataset_name, sep='\t', index=False)
raw_df_common.to_csv('../data/%s/semi_1.5/divided/train_common.csv' %dataset_name, sep='\t', index=False)

print('Filtering cross-dataset facts ...')

raw_ent1 = set(raw_df1['head'].values).union(set(raw_df1['tail'].values))
raw_ent2 = set(raw_df2['head'].values).union(set(raw_df2['tail'].values))
raw_ent_common = set(raw_df_common['head'].values).union(set(raw_df_common['tail'].values))

print('raw df1 ent num: ', len(raw_ent1))
print('raw df2 ent num: ', len(raw_ent2))
print('raw df_common ent num: ', len(raw_ent_common))

print('overlap percentage: %.5f' %(len(raw_ent_common) / len(raw_ent1)))


def collect_cross_facts(raw_ent1, raw_ent2):
    """
    collect cross facts (h, r, t) s.t h in raw_ent1, t in raw_ent2 and r in common_raw_rel
    """
    data = []
    for h, r, t in zip(head_raw, rel_raw, tail_raw):
        if (h in raw_ent1) and (t in raw_ent2) and (r in common_raw_rel):
            data.append([h, r, t])
        elif (h in raw_ent2) and (t in raw_ent1) and (r in common_raw_rel):
            data.append([h, r, t])

    return pd.DataFrame(data, columns=df_names)

facts_12 = collect_cross_facts(raw_ent1, raw_ent2)
print('Done collect facts between 1 and 2')
facts_1_common = collect_cross_facts(raw_ent1, raw_ent_common)
print('Done collect facts between common and 1')
facts_2_common = collect_cross_facts(raw_ent2, raw_ent_common)
print('Done collect facts between common and 2')


print('Facts 12: ', facts_12.shape)
print('Facts 1_common: ', facts_1_common.shape)
print('Facts 2_common: ', facts_2_common.shape)


print('Saving cross-data facts ...')

facts_12.to_csv('../data/%s/semi_1.5/divided/cross_12.csv' %dataset_name, sep='\t', index=False)

facts_1_common.to_csv('../data/%s/semi_1.5/divided/cross_1_common.csv' %dataset_name, sep='\t', index=False)

facts_2_common.to_csv('../data/%s/semi_1.5/divided/cross_2_common.csv' %dataset_name, sep='\t', index=False)







