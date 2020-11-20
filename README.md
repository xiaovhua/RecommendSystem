# RecommendSystem
项亮《推荐系统实践》的具体实现


主要参考内容为书本《推荐系统实践》本身，以及 qcymkxyc 大大的 github 代码 https://github.com/qcymkxyc/RecSys 和 Magic-Bubble 的 github 代码 https://github.com/Magic-Bubble/RecommendSystemPractice


代码组织结构如下：
1、在主目录下的两个文件，在各个章节中都可能会用到。他们是：
utils.py 存放工具函数，如读取数据、数据分割（用于交叉验证）等
metrics.py 存放评价指标，如召回率、准确率、覆盖率和新颖度（用总的流行度衡量）
2、Chapter2 存放第二章的代码
usercf.py  useriif.py  itemcf.py  itemiuf.py  itemcfNorm.py  lfm.py  personalRank.py  分别存放封装好的协同过滤算法  
main.py  是各算法在 MovieLen 数据集上的实现




