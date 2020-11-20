from Chapter2.itemcf import itemcf


class itemcfNorm(itemcf):
    def item_similarity(self):
        item_sim_matrix = itemcf.item_similarity(self)
        for u, related_items in item_sim_matrix.items():
            _max = max(related_items.values())
            for v, sim in related_items.items():
                item_sim_matrix[u][v] /= _max
        return item_sim_matrix

