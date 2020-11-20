import numpy as np
import collections
import tqdm

class lfm():
    def __init__(self, ratio, alpha=0.02, regularization_rate=0.01, F=100):
        self.ratio = ratio
        self.embedding_dim = F
        self.learning_rate = alpha
        self.regularization_rate = regularization_rate

    def train(self, training_data, epoch):
        """
        :param training_data: {user1: {item1, item4}, user2: {item3}, ...}
        :param epoch:
        :return:
        """
        self.training_data = training_data
        self._getPopularity()
        self._init()
        lr = self.learning_rate
        for i in tqdm.tqdm(range(epoch)):
            # 负采样
            data = self._sekectNegativeSample(self.ratio)
            for user, items in data.items():
                for item, rating in items.items():
                    # 前向传播
                    eui = rating - self._predict(user, item)
                    # 更新
                    self.p[user] += lr * (eui * self.q[item] - self.regularization_rate * self.p[user])
                    self.q[item] += lr * (eui * self.p[user] - self.regularization_rate * self.q[item])
            if (i + 1) % 100 == 0:
                print('loss: %f'%(self._loss()))
            lr *= 0.9

    def _init(self):
        self.p = collections.defaultdict(lambda: np.random.random(size=(self.embedding_dim,)))
        self.q = collections.defaultdict(lambda: np.random.random(size=(self.embedding_dim,)))

    def _getPopularity(self):
        """
        获取各物品流行度
        :return:
        """
        all_items = collections.defaultdict(int)
        for user, items in self.training_data.items():
            for item in items:
                all_items[item] += 1
        self.items = [x[0] for x in all_items.items()]
        self.popularities = [x[1] for x in all_items.items()]

    def _sekectNegativeSample(self, ratio):
        """
        负样本采样
        :param ratio: 负样本/正样本的比例
        :return:
        """
        new_data = collections.defaultdict(dict)
        # 正样本
        for user, items in self.training_data.items():
            for item in items:
                new_data[user][item] = 1
        # 负样本
        for user in new_data:
            interacted_items = set(new_data[user].keys())
            positive_num = len(interacted_items)
            # int(positive_num * (ratio + 1) 保证去重后数量能满足 positive_num * ratio
            try:
                item = np.random.choice(self.items, int(positive_num * (ratio + 1)), replace=False, p=np.array(self.popularities)/sum(self.popularities))
            except:
                item = self.items[:]
            # 去重
            item = [x for x in item if x not in interacted_items][:int(positive_num * ratio)]
            new_data[user].update({x: 0 for x in item})
        return new_data

    def _predict(self, user, item):
        return np.dot(self.p[user], self.q[item])

    def _loss(self):
        C = 0
        for user, user_p in self.p.items():
            for item, item_q in self.q.items():
                if item in self.training_data[user]:
                    rui = 1
                else:
                    rui = 0
                eui = rui - self._predict(user, item)
                C += eui ** 2 + self.regularization_rate * (np.linalg.norm(self.p[user]) ** 2 + np.linalg.norm(self.q[item]) ** 2)
        return C

    def recommend(self, user):
        interacted_items = set(self.training_data[user])
        recommend = dict()
        for item, q in self.q.items():
            if item not in interacted_items:
                recommend[item] = self._predict(user, item)
        return sorted(recommend.items(), key=lambda x: x[1], reverse=True)

    def recommends(self, users, N):
        recommends = dict()
        for user in users:
            recommends[user] = [x[0] for x in self.recommend(user)][:N]
        return recommends

