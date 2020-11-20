from Chapter2 import usercf, useriif, itemcf, itemiuf, itemcfNorm, lfm, personalRank
from utils import load, getDict, splitData, getAllItem, getItemPopularity
from metrics import Recall, Precision, Coverage, Popilarity
import tqdm

seed = 23
N = 10
M = 8
if __name__ == '__main__':
    # # 1. 简单的例子
    # training = {
    #     'A': {'a', 'b', 'd'},
    #     'B': {'a', 'c'},
    #     'C': {'b', 'e'},
    #     'D': {'c', 'd', 'e'}
    # }
    # usercf = usercf.usercf()  # useriif.useriif()
    # usercf.train(training_data=training, save_path='D:/PyProject/项亮《推荐系统时间》/Chapter2/data/toyExample.pkl')
    # recommends = usercf.recommends(users=training.keys(), K=3, N=2)
    # print(recommends)



    # # 2. movielen-1m, usercf / useriif / itemcf

    # # usercf
    # K = 80

    # # itemcf
    # K = 10

    # data = load(data_path='D:\\PyProject\\项亮《推荐系统时间》\\ml-1m')
    # recall_all = precision_all = coverage_all = popularity_all = 0
    # all_items, item_popularity = getAllItem(data), getItemPopularity(data)
    # # model = usercf.usercf()
    # # model = useriif.useriif()
    # # model = itemcf.itemcf()
    # # model = itemiuf.itemiuf()
    # model = itemcfNorm.itemcfNorm()
    # for i in range(M):
    #     train, test = splitData(data, M=M, i=i, seed=seed)
    #     train, test = getDict(train), getDict(test)
    #     model.train(training_data=train, save_path='D:/PyProject/项亮《推荐系统时间》/Chapter2/data/itemnorm_%d.pkl'%(i + 1))
    #     recommends = model.recommends(users=test.keys(), K=K, N=N)
    #     recall, precision, coverage, popularity = \
    #         Recall(recommends, test), Precision(recommends, test), Coverage(recommends, all_items), Popilarity(recommends, item_popularity)
    #     print('Fold %d, Recall: %f, Precision: %f, Coverage: %f, Popilarity: %f'%(i + 1, recall, precision, coverage, popularity))
    #     recall_all += recall
    #     precision_all += precision
    #     coverage_all += coverage
    #     popularity_all += popularity
    #     print('\n\n')
    # print('Recall: %f, Precision: %f, Coverage: %f, Popilarity: %f'%(recall_all/M, precision_all/M, coverage_all/M, popularity_all/M))



    # # 3. lfm
    # data = load(data_path='D:\\PyProject\\项亮《推荐系统时间》\\ml-1m')
    # recall_all = precision_all = coverage_all = popularity_all = 0
    # all_items, item_popularity = getAllItem(data), getItemPopularity(data)
    # model = lfm.lfm(alpha=0.02, regularization_rate=0.01, F=100, ratio=10)
    # for i in range(M):
    #     train, test = splitData(data, M=M, i=i, seed=seed)
    #     train, test = getDict(train), getDict(test)
    #     model.train(training_data=train, epoch=100)
    #     recommends = model.recommends(users=test.keys(), N=N)
    #     recall, precision, coverage, popularity = \
    #         Recall(recommends, test), Precision(recommends, test), Coverage(recommends, all_items), Popilarity(recommends, item_popularity)
    #     print('Fold %d, Recall: %f, Precision: %f, Coverage: %f, Popilarity: %f'%(i + 1, recall, precision, coverage, popularity))
    #     recall_all += recall
    #     precision_all += precision
    #     coverage_all += coverage
    #     popularity_all += popularity
    #     print('\n\n')
    # print('Recall: %f, Precision: %f, Coverage: %f, Popilarity: %f'%(recall_all/M, precision_all/M, coverage_all/M, popularity_all/M))


    # # 4. personal rank
    data = load(data_path='D:\\PyProject\\项亮《推荐系统时间》\\ml-1m')[:10000]
    recall_all = precision_all = coverage_all = popularity_all = 0
    all_items, item_popularity = getAllItem(data), getItemPopularity(data)
    model = personalRank.personalRank(alpha=0.8)
    for i in range(M):
        train, test = splitData(data, M=M, i=i, seed=seed)
        train, test = getDict(train), getDict(test)
        model.buildGraph(training_data=train)
        recommends = model.recommends(users=test.keys(), N=N)
        recall, precision, coverage, popularity = \
            Recall(recommends, test), Precision(recommends, test), Coverage(recommends, all_items), Popilarity(recommends, item_popularity)
        print('Fold %d, Recall: %f, Precision: %f, Coverage: %f, Popilarity: %f'%(i + 1, recall, precision, coverage, popularity))
        recall_all += recall
        precision_all += precision
        coverage_all += coverage
        popularity_all += popularity
        print('\n\n')
    print('Recall: %f, Precision: %f, Coverage: %f, Popilarity: %f'%(recall_all/M, precision_all/M, coverage_all/M, popularity_all/M))

