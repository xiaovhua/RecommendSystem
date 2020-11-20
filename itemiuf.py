from Chapter2.itemcf import itemcf
import collections
import math


class itemiuf(itemcf):
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
                        item_sim_matrix[u][v] += 1 / math.log(1 + len(items)) # 唯一的不同
        # 计算物品相似度矩阵
        for u, related_items in item_sim_matrix.items():
            for v, count in related_items.items():
                item_sim_matrix[u][v] /= math.sqrt(N[u] * N[v])
        return item_sim_matrix

