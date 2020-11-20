import random
import math
import collections
import pandas as pd
import os

def splitData(data, M, i, seed):
    """
    M 折交叉验证，取第 i 折
    :param data: [(user1, item1), (user2, item2), ...]
    :param M:
    :param i:
    :param seed:
    :return:
    """
    test = []
    train = []
    random.seed(seed)
    for user, item, rating in data:
        if random.randint(0, M - 1) == i:
            test.append((user, item, rating))
        else:
            train.append((user, item, rating))
    return train, test


def load(data_path):
    table = pd.read_table(os.path.join(data_path, 'ratings.dat'), sep='::', header=None).drop(columns=3)
    data = []
    for index, rows in table.iterrows():
        user, item, rating = rows
        data.append((user, item, rating))
    return data


def getDict(dataset):
    d = collections.defaultdict(set)
    for user, item, rating in dataset:
         d[user].add(item)
    return d


def getAllItem(dataset):  # 计算覆盖率
    all_items = set()
    for user, item, rating in dataset:
        all_items.add(item)
    return all_items


def getItemPopularity(dataset):  # 计算新颖性
    item_popularity = collections.defaultdict(int)
    for user, item, rating in dataset:
        item_popularity[item] += 1
    return item_popularity


#  -------------------------------------------------   未使用  ----------------------------------------------
def CosSimilarity(train):
    """
    :param train: {user1: {item1: rating, item2: rating, ...}, ...}
    :return:
    """
    W = collections.defaultdict(dict)
    for u in train.keys():
        for v in train.keys():
            if u != v:
                if v not in W[u]:
                    W[u][v] = 0
                W[u][v] = len(train[u] & train[v])
                W[u][v] /= math.sqrt(len(train[u]) * len(train[v]))
    return W


def JaccardSimilarity(train):
    """
    :param train: {user1: {item1: rating, item2: rating, ...}, ...}
    :return:
    """
    W = collections.defaultdict(dict)
    for u in train.keys():
        for v in train.keys():
            if u != v:
                if v not in W[u]:
                    W[u][v] = 0
                W[u][v] = len(train[u] & train[v])
                W[u][v] /= math.sqrt(len(train[u]) | len(train[v]))
    return W
