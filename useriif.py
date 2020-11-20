from Chapter2.usercf import usercf
import collections
import math

class useriif(usercf):
    def user_similarity(self):
        # 建立倒排表
        item_users = collections.defaultdict(set)
        for user, items in self.training.items():
            for item in items:
                item_users[item].add(user)
        # 计算协同过滤矩阵
        user_sim_matrix = collections.defaultdict(dict)
        N = collections.defaultdict(int)
        for item, users in item_users.items():
            for u in users:
                N[u] += 1
                for v in users:
                    if u != v:
                        user_sim_matrix[u].setdefault(v, 0.0)
                        user_sim_matrix[u][v] += 1 / math.log(1 + len(users))  # 唯一的区别
        # 计算用户相似度矩阵
        for u, related_users in user_sim_matrix.items():
            for v, count in related_users.items():
                user_sim_matrix[u][v] /= math.sqrt(N[u] * N[v])
        return user_sim_matrix

