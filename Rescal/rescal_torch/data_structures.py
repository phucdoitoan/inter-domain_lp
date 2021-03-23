

import torch
from torch.utils.data import Dataset
import pandas as pd


def array2dict(arr):
    """
    Args:
        arr: array like
    Returns:
        dict_id: dict that maps data in arr to integer index
    """
    dict_id = {}
    for i in arr:
        if i not in dict_id.keys():
            dict_id[i] = len(dict_id)

    return dict_id

class KnowledgeGraph(Dataset):
    """
    Knowledge graph representation
    Parameters:
        df: pandas.DataFrame, optional
            containing three columns [head, rel, tail]
        kg: dict with keys of 'head', 'rel', 'tail' and
            values of corresponding torch long tensors
        ent2id: dict: entities to integer idx
        rel2id: dict: relations to integer idx
    """

    def __init__(self, df=None, kg=None, ent2id=None, rel2id=None):

        if df is not None:
            self.df = df
            if ent2id is not None:
                self.ent2id = ent2id
            else:
                self.ent2id = array2dict(pd.concat([self.df['head'], self.df['tail']]))

            if rel2id is not None:
                self.rel2id = rel2id
            else:
                self.rel2id = array2dict(self.df['rel'].values)

            self.n_ent = len(self.ent2id)
            self.n_rel = len(self.rel2id)
            self.n_facts = len(df)

            self.head_idx = torch.LongTensor(self.df['head'].map(self.ent2id).values)
            self.tail_idx = torch.LongTensor(self.df['tail'].map(self.ent2id).values)
            self.relations = torch.LongTensor(self.df['rel'].map(self.rel2id).values)

            print('Created a KnowledgeGraph with dataframe provided!')

        elif kg is not None:
            self.kg = kg
            if ent2id is not None:
                self.ent2id = ent2id
            else:
                self.ent2id = array2dict(torch.cat((self.kg['head'], self.kg['tail']), dim=0))

            if rel2id is not None:
                self.rel2id = rel2id
            else:
                self.rel2id = array2dict(kg['rel'])

            self.n_ent = len(self.ent2id)
            self.n_rel = len(self.rel2id)
            self.n_facts = self.kg['head'].shape[0]
            assert self.n_facts == self.kg['tail'].shape[0] and self.n_facts == self.kg['rel'].shape[0], \
                'head, tail, and rel of kg must be of equal length: head-%s tail-%s rel-%s' %(self.kg['head'].shape[0], self.kg['tail'].shape[0], self.kg['rel'].shape[0])

            self.head_idx = self.kg['head']
            self.tail_idx = self.kg['tail']
            self.relations = self.kg['rel']

            print('Created a KnowledgeGraph with a dictionary provided!')

        else:
            raise Exception('At least one of the arguments df and kg must not be None: df-%s kg-%s ' %(df, kg))

        # initialize dict_of_tails and dict_of_heads
        self.dict_of_tails = {}
        self.dict_of_heads = {}

        for i in range(self.n_facts):
            if (self.head_idx[i].item(), self.relations[i].item()) in self.dict_of_tails.keys():
                self.dict_of_tails[self.head_idx[i].item(), self.relations[i].item()].append(self.tail_idx[i].item())
            else:
                self.dict_of_tails[self.head_idx[i].item(), self.relations[i].item()] = [self.tail_idx[i].item()]

            if (self.tail_idx[i].item(), self.relations[i].item()) in self.dict_of_heads.keys():
                self.dict_of_heads[self.tail_idx[i].item(), self.relations[i].item()].append(self.head_idx[i].item())
            else:
                self.dict_of_heads[self.tail_idx[i].item(), self.relations[i].item()] = [self.head_idx[i].item()]

    def __len__(self):
        return self.n_facts

    def __getitem__(self, item):
        return (self.head_idx[item].item(),
                self.tail_idx[item].item(),
                self.relations[item].item())

    # No need for splitting the knowledge graphs because the available datasets are already divivded into train, valid, test data
    def split_kg(self, size):
        """
        split the knowledge graph into train, valid and test (depend on the variable size)
        Args:
            size: tuple like of spliting ratio
                  if length of size is 1 -> split kg into train and test kg
                  if the length is 2 -> split kg into train, valid, and test kg
        Returns:
            train_kg
            valid_kg (optional)
            test_kg
        """
        idx = torch.randperm(self.n_facts)

        if len(size) == 1:
            train_size = int(self.n_facts * size[0])
            train_idx = idx[:train_size]
            test_idx = idx[train_size:]
            return (
                KnowledgeGraph(
                    kg = {
                        'head': self.head_idx[train_idx],
                        'tail': self.tail_idx[train_idx],
                        'rel': self.relations[train_idx]
                    },
                    ent2id = self.ent2id,
                    rel2id = self.rel2id,
                ),
                KnowledgeGraph(
                    kg={
                        'head': self.head_idx[test_idx],
                        'tail': self.tail_idx[test_idx],
                        'rel': self.relations[test_idx]
                    },
                    ent2id=self.ent2id,
                    rel2id=self.rel2id,
                )
            )
        else:
            train_size = int(self.n_facts * size[0])
            valid_size = int(self.n_facts * size[1])

            train_idx = idx[:train_size]
            valid_idx = idx[train_size: train_size + valid_size]
            test_idx = idx[train_size + valid_size:]
            return (
                KnowledgeGraph(
                    kg={
                        'head': self.head_idx[train_idx],
                        'tail': self.tail_idx[train_idx],
                        'rel': self.relations[train_idx]
                    },
                    ent2id=self.ent2id,
                    rel2id=self.rel2id,
                ),
                KnowledgeGraph(
                    kg={
                        'head': self.head_idx[valid_idx],
                        'tail': self.tail_idx[valid_idx],
                        'rel': self.relations[valid_idx]
                    },
                    ent2id=self.ent2id,
                    rel2id=self.rel2id,
                ),
                KnowledgeGraph(
                    kg={
                        'head': self.head_idx[test_idx],
                        'tail': self.tail_idx[test_idx],
                        'rel': self.relations[test_idx]
                    },
                    ent2id=self.ent2id,
                    rel2id=self.rel2id,
                ),
            )




