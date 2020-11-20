import collections
import sys
import os
import pickle
import math

class itemcf():
    def train(self, training_data, save_path):
        self.training = training_data
        try:
            print('开始载入物品相似度矩阵', file=sys.stderr)
            with open(save_path, 'rb') as f:
                self.item_sim_matrix = pickle.load(f)
            print('载入物品相似度矩阵成功', file=sys.stderr)
        except:
            print('载入物品相似度矩阵失败，重新计算', file=sys.stderr)
            self.item_sim_matrix = self.item_similarity()
            print('正在保存物品相似度矩阵', file=sys.stderr)
            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.mkdir(save_path[:save_path.rfind('/')])
            with open(save_path, 'wb') as f:
                pickle.dump(self.item_sim_matrix, f)
            print('物品相似度矩阵保存成功', file=sys.stderr)


    def item_similarity(self):
        # 建立倒排表
        user_items = self.training
        # 计算协同过滤矩阵
        item_sim_matrix = collections.defaultdict(dict)
        N = collections.defaultdict(int)
        for user, items in user_items.items():
            for u in items:
                N[u] += 1
                for v in items:
                    if u != v:
                        item_sim_matrix[u].setdefault(v, 0)
                        item_sim_matrix[u][v] += 1
        # 计算物品相似度矩阵
        for u, related_items in item_sim_matrix.items():
            for v, count in related_items.items():
                item_sim_matrix[u][v] /= math.sqrt(N[u] * N[v])
        return item_sim_matrix


    def recommend(self, user, K, N):
        interacted_items = self.training[user]
        recommend = collections.defaultdict(float)
        for u in self.training[user]:
            for v, sim in sorted(self.item_sim_matrix[u].items(), key=lambda x: x[1], reverse=True)[:K]:
                if v not in interacted_items:
                    recommend[v] += sim
        return dict(sorted(recommend.items(), key=lambda x: x[1], reverse=True)[:N])


    def recommends(self, users, K, N):
        recommends = dict()
        for user in users:
            recommends[user] = self.recommend(user, K, N)
        return recommends
