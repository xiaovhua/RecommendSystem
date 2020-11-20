import math


def RMSE(records):
    """
    :param records: [(pred_raating, real_rating), (pred_raating, real_rating), ...]
    :return:
    """
    numerator = sum([(pred_rating - real_rating) ** 2 for pred_rating, real_rating in records])
    denominator = float(len(records))
    return math.sqrt(numerator / denominator)


def MAE(records):
    """
    :param records: [(pred_raating, real_rating), (pred_raating, real_rating), ...]
    :return:
    """
    numerator = sum([abs(pred_rating - real_rating) for pred_rating, real_rating in records])
    denominator = float(len(records))
    return numerator / denominator


def Recall(recommends, test):
    """
    :param recommends: {uid1: [item, item3], ...}
    :param test: {uid1: [item1, item2], ...}
    :return:
    """
    hit = 0
    all = 0
    for user, items in recommends.items():
        pred = set(items)
        real = set(test[user])
        hit += len(pred & real)
        all += len(real)
    return hit / all * 100


def Precision(recommends, test):
    """
    :param recommends: {uid1: [item, item3], ...}
    :param test: {uid1: [item1, item2], ...}
    :return:
    """
    hit = 0
    all = 0
    for user, items in recommends.items():
        pred = set(items)
        real = set(test[user])
        hit += len(pred & real)
        all += len(pred)  # the only difference
    return hit / all * 100


def Coverage(recommends, all_items):
    """
    :param recommends: {uid1: [item, item3], ...}
    :param all_items: {item1, item2, ....}
    :return:
    """
    recommend_items = set()
    for user, items in recommends.items():
        recommend_items.update(items)
    return len(recommend_items) / len(all_items) * 100


def Popilarity(recommends, item_popularity):
    """
    :param recommends: {uid1: [item, item3], ...}
    :param item_popularity: {item1: count, item2: count, ...}
    :return:
    """
    popularity = 0
    n = 0
    for user, items in recommends.items():
        for item in items:
            popularity += math.log(1 + item_popularity.get(item, 0.0))
            n += 1
    return popularity / n
