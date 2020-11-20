import collections
import sys
import os
import pickle
import math

class usercf():
    def train(self, training_data, save_path):
        self.training = training_data
        try:
            print('开始载入用户相似度矩阵', file=sys.stderr)
            with open(save_path, 'rb') as f:
                self.user_sim_matrix = pickle.load(f)
            print('载入用户相似度矩阵完成', file=sys.stderr)
        except:
            print('载入用户相似度矩阵失败，重新计算', file=sys.stderr)
            self.user_sim_matrix = self.user_similarity()
            print('开始保存用户相似度矩阵', file=sys.stderr)
            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.mkdir(save_path[:save_path.rfind('/')])
            with open(save_path, 'wb') as f:
                pickle.dump(self.user_sim_matrix, f)
            print('保存用户相似度矩阵成功', file=sys.stderr)

    def user_similarity(self):
        # 建立倒排表
        item_users = collections.defaultdict(set)
        for user, items in self.training.items():
            for item in items:
                item_users[item].add(user)
        # 计算协同过滤矩阵
        user_sim_matrix = collections.defaultdict(dict)
        N = collections.defaultdict(int)  # 计算用户购买的商品数
        for item, users in item_users.items():
            for u in users:
                N[u] += 1
                for v in users:
                    if u != v:
                        user_sim_matrix[u].setdefault(v, 0)
                        user_sim_matrix[u][v] += 1
        # 计算用户相似度矩阵
        for u, related_users in user_sim_matrix.items():
            for v, count in related_users.items():
                user_sim_matrix[u][v] /= math.sqrt(N[u] * N[v])
        return user_sim_matrix

    def recommend(self, user, N, K):
        """
        :param user:
        :param N: 推荐商品推荐度最高的 N 个商品
        :param K: 选择 K 个相关性最高的用户计算商品推荐度
        :return:
        """
        interacted_items = self.training[user]  # 已购买的不再推荐
        recommend = collections.defaultdict(float)
        # 由用户相关性计算商品推荐程度
        for v, sim in sorted(self.user_sim_matrix[user].items(), key=lambda x: x[1], reverse=True)[:K]:
            for item in self.training[v]:
                if item not in interacted_items:
                    recommend[item] += sim
        return dict(sorted(recommend.items(), key=lambda x: x[1], reverse=True)[:N])

    def recommends(self, users, N, K):
        recommends = dict()
        for user in users:
            recommends[user] = self.recommend(user, N, K)
        return recommends

