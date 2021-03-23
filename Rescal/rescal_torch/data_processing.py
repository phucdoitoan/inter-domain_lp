

import pandas as pd
from data_structures import KnowledgeGraph
import torch
import random
from time import time
import os


df_names = ['head', 'rel', 'tail']

#dataset_name = 'FB15k-237'
dataset_name = 'WN18RR'

train_df = pd.read_csv('../data/%s/train.txt' %dataset_name, delimiter='\t', header=None, names=df_names)
valid_df = pd.read_csv('../data/%s/valid.txt' %dataset_name, delimiter='\t', header=None, names=df_names)
test_df = pd.read_csv('../data/%s/test.txt' %dataset_name, delimiter='\t', header=None, names=df_names)

df = pd.concat([train_df, valid_df, test_df], axis=0)

raw_ent = list(set(df['head'].values).union(set(df['tail'].values)))
print('raw_ent: ', len(raw_ent))

random.shuffle(raw_ent)

ent_size = 10000 

raw_ent1 = raw_ent[:ent_size]
raw_ent2 = raw_ent[ent_size: 2*ent_size]
raw_ent3 = raw_ent[2*ent_size: 3*ent_size]

print('raw_ent1: ', len(raw_ent1))
print('raw_ent2: ', len(raw_ent2))
print('raw_ent3: ', len(raw_ent3))

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
print('done 1: %.4f s' %(time() - t0))
raw_df2 = raw_ent_filter(df, raw_ent2)
print('done 2: %.4f s' %(time() - t0))
raw_df3 = raw_ent_filter(df, raw_ent3)
print('done 3: %.4f s' %(time() - t0))

print('raw_df1: ', raw_df1.shape)
print('raw_df2: ', raw_df2.shape)
print('raw_df3: ', raw_df3.shape)

print('REL number df1: ', len(set(raw_df1['rel'].values)))
print('REL number df2: ', len(set(raw_df2['rel'].values)))
print('REL number df3: ', len(set(raw_df3['rel'].values)))

print('Adjusting so that all data share a same set of relations')

common_raw_rel = set(raw_df1['rel'].values).intersection(set(raw_df2['rel'].values)).intersection(set(raw_df3['rel'].values))
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
raw_df3 = unify_rel(raw_df3, separate_raw_rel)
print('done unify_rel 3: %.4f s' %(time() - t0))


print('REL number df1: ', len(set(raw_df1['rel'].values)))
print('REL number df2: ', len(set(raw_df2['rel'].values)))
print('REL number df3: ', len(set(raw_df3['rel'].values)))

print('Relation sets of df1, df2, df3 is the same: ', set(raw_df1['rel'].values) == set(raw_df2['rel'].values) == set(raw_df3['rel'].values))

print('Saving all dataframes into csv files ...')

if not os.path.exists('../data/%s/divided/' %dataset_name):
    os.makedirs('../data/%s/divided/' %dataset_name)

raw_df1.to_csv('../data/%s/divided/train1.csv' %dataset_name, sep='\t', index=False)
raw_df2.to_csv('../data/%s/divided/train2.csv' %dataset_name, sep='\t', index=False)
raw_df3.to_csv('../data/%s/divided/train3.csv' %dataset_name, sep='\t', index=False)

print('Filtering cross-dataset facts ...')

raw_ent1 = set(raw_df1['head'].values).union(set(raw_df1['tail'].values))
raw_ent2 = set(raw_df2['head'].values).union(set(raw_df2['tail'].values))
raw_ent3 = set(raw_df3['head'].values).union(set(raw_df3['tail'].values))

print('raw df1 ent num: ', len(raw_ent1))
print('raw df2 ent num: ', len(raw_ent2))
print('raw df3 ent num: ', len(raw_ent3))


def collect_cross_facts(raw_ent1, raw_ent2):
    """
    collect cross facts (h, r, t) s.t h in raw_ent1, t in raw_ent2 and r in common_raw_rel
    """
    data = []
    for h, r, t in zip(head_raw, rel_raw, tail_raw):
        if (h in raw_ent1) and (t in raw_ent2) and (r in common_raw_rel):
            data.append([h, r, t])

    return pd.DataFrame(data, columns=df_names)

facts_h1t2 = collect_cross_facts(raw_ent1, raw_ent2)
facts_h2t1 = collect_cross_facts(raw_ent2, raw_ent1)
print('Done collect facts 12: ')
facts_h1t3 = collect_cross_facts(raw_ent1, raw_ent3)
facts_h3t1 = collect_cross_facts(raw_ent3, raw_ent1)
print('Done collect facts 13: ')
facts_h2t3 = collect_cross_facts(raw_ent2, raw_ent3)
facts_h3t2 = collect_cross_facts(raw_ent3, raw_ent2)
print('Done collect facts 23: ')


print('Facts 12: ', facts_h1t2.shape, facts_h2t1.shape)
print('Facts 13: ', facts_h1t3.shape, facts_h3t1.shape)
print('Facts 23: ', facts_h2t3.shape, facts_h3t2.shape)


print('Saving cross-data facts ...')

facts_h1t2.to_csv('../data/%s/divided/cross_h1t2.csv' %dataset_name, sep='\t', index=False)
facts_h2t1.to_csv('../data/%s/divided/cross_h2t1.csv' %dataset_name, sep='\t', index=False)

facts_h1t3.to_csv('../data/%s/divided/cross_h1t3.csv' %dataset_name, sep='\t', index=False)
facts_h3t1.to_csv('../data/%s/divided/cross_h3t1.csv' %dataset_name, sep='\t', index=False)

facts_h2t3.to_csv('../data/%s/divided/cross_h2t3.csv' %dataset_name, sep='\t', index=False)
facts_h3t2.to_csv('../data/%s/divided/cross_h3t2.csv' %dataset_name, sep='\t', index=False)



print('df: ', df.shape)










"""
kg = KnowledgeGraph(df=df)
print('kg n_ent: %s, n_rel: %s, n_facts: %s'  %(kg.n_ent, kg.n_rel, kg.n_facts))


#train_kg, valid_kg, test_kg = kg.split_kg(size=(0.4, 0.3))
#print('train_kg n_ent: %s, n_rel: %s, n_facts: %s'  %(train_kg.n_ent, train_kg.n_rel, train_kg.n_facts))
#print('valid_kg n_ent: %s, n_rel: %s, n_facts: %s'  %(valid_kg.n_ent, valid_kg.n_rel, valid_kg.n_facts))
#print('test_kg n_ent: %s, n_rel: %s, n_facts: %s'  %(test_kg.n_ent, test_kg.n_rel, test_kg.n_facts))
#print('test_kg.ent2id: ', test_kg.rel2id)



indexes = torch.randperm(kg.n_ent)

ent_size = 5000
idx_ent1 = indexes[:ent_size]
idx_ent2 = indexes[ent_size: 2*ent_size]
idx_ent3 = indexes[2*ent_size: 3*ent_size]



def facts_filter(kg, idx_ent):
    #Filter facts from a knowledge graph, in which entities in idx_ent appear in (as head or tail)
    print('idx_ent: ', idx_ent[:10])
    ent_set = set(idx_ent.tolist())

    print('ent_set: ', len(ent_set))

    data = []

    for h, t, r in zip(kg.head_idx, kg.tail_idx, kg.relations):
        h, t, r = h.item(), t.item(), r.item()

        #if (h in ent_set) or (t in ent_set):
        #    data.append([h, r, t])
        if (h in ent_set) and (t in ent_set):
            data.append([h, r, t])

    df = pd.DataFrame(data, columns=['head', 'rel', 'tail'])

    return df

df1 = facts_filter(kg, idx_ent1)
df2 = facts_filter(kg, idx_ent2)
df3 = facts_filter(kg, idx_ent3)

print('df1: ', df1.shape)
print('df2: ', df2.shape)
print('df3: ', df3.shape)

common_rel = set(df1['rel'].values.tolist()).intersection(set(df2['rel'].values.tolist())).intersection(set(df3['rel'].values.tolist()))

print('relations intersection of df1 df2 df3: ', len(common_rel))
#print('common relations: ', common_rel)

separate_rel = set(kg.relations.tolist()).difference(common_rel)

print('len kg.df[rel] ', len(set(kg.relations.tolist())))

print('separate relations: ', separate_rel)

def drop_row(df, separate_rel):
    #drop row of df if rel is in separate_rel
    for r in separate_rel:
        df = df.drop(df[df['rel']==r].index)

    return df

df1 = drop_row(df1, separate_rel)
df2 = drop_row(df2, separate_rel)
df3 = drop_row(df3, separate_rel)



kg1 = KnowledgeGraph(df1)
kg2 = KnowledgeGraph(df2)
kg3 = KnowledgeGraph(df3)

print('kg1 n_ent: %s, n_rel: %s, n_facts: %s'  %(kg1.n_ent, kg1.n_rel, kg1.n_facts))
print('kg2 n_ent: %s, n_rel: %s, n_facts: %s'  %(kg2.n_ent, kg2.n_rel, kg2.n_facts))
print('kg3 n_ent: %s, n_rel: %s, n_facts: %s'  %(kg3.n_ent, kg3.n_rel, kg3.n_facts))

rel_kg1 = set(kg1.df['rel'].values)
rel_kg2 = set(kg2.df['rel'].values)
rel_kg3 = set(kg3.df['rel'].values)

print('rel_kg123 is the same: ', rel_kg1 == rel_kg2 == rel_kg3)
"""











