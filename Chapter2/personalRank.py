import numpy as np
from scipy.sparse import csc_matrix, linalg, eye


class personalRank():
    def __init__(self, alpha):
        self.alpha = alpha

    def buildGraph(self, training_data):
        self.training_data = training_data
        all_items = set()
        for user in training_data:
            all_items.update(training_data[user])
        self.all_items = list(all_items)
        # 给用户和物品编号作为行列索引
        self.users = {u: i for i, u in enumerate(training_data.keys())}
        self.items = {u: i + len(self.users) for i, u in enumerate(self.all_items)}
        # 计算 1.用户:{物品}倒排表 training_data  2.物品:{用户}倒排表 item_users
        item_users = {}
        for user in training_data:
            for item in training_data[user]:
                item_users.setdefault(item, [])
                item_users[item].append(user)
        # 根据倒排表 training_data 和 item_users 每一项的度，构造 (|U| + |I|) * (|U| + |I|) 维稀疏矩阵 M
        data, row, col = [], [], []  # 数据、行索引、列索引
        for u in training_data:
            for v in training_data[u]:
                data.append(1 / len(training_data[u]))
                row.append(self.users[u])
                col.append(self.items[v])
        for u in item_users:
            for v in item_users[u]:
                data.append(1 / len(item_users[u]))
                row.append(self.items[u])
                col.append(self.users[v])

        self.M = csc_matrix((data, (row, col)), shape=(len(data), len(data)))

    def recommend(self, user, N):
        interacted_items = set(self.training_data[user])
        # 解矩阵方程:
        # r = (1 - a)r0 + a(M.T)r  ->  r = (1 - a) * r0 * (E - a(M.T))-1, r.T = (1 - a) * (E - a(M.T))-1 * r0
        r0 = [0] * self.M.shape[0]  # r0 = (|U| + |I|, )，表示每一个顶点（用户+物品）的重要度
        r0[self.users[user]] = 1  # 找到用户为 user 一行，将 user 对应的 r 设置为 1，其余为 0
        r0 = csc_matrix(r0)  # 将该一维向量转为稀疏向量

        # r0: (1, |U| + |I|), M: (|U| + |I|, |U| + |I|)
        r = (1 - self.alpha) * np.dot(r0, linalg.inv(eye(self.M.shape[0]) - self.alpha * self.M.T))
        r = r.T.toarray()[0][len(self.users):]  # 只取后面 |I| 个物品的重要度
        idx = np.argsort(-r)  # 排序并保存下标
        recommend = [(self.all_items[i], r[i]) for i in idx if self.all_items[i] not in interacted_items][:N]  # 推荐前 N 个
        return recommend

    def recommends(self, users, N):
        recommends = dict()
        for user in users:
            rank = self.recommend(user, N)
            recommends[user] = [x[0] for x in rank]
        return recommends
