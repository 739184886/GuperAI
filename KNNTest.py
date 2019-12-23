# import numpy as np
# import matplotlib.pyplot as plt
# ####
# ####聚类算法
# # 标签数据
# group = np.array([
#     [1.0,1.1],
#     [1.0,1.0],
#     [0,0],
#     [0,0.1]
# ])
#
# # 标签
# lables = ['A','A','B','B']
#
# # 分类点展示
# def show_data(group,lables):
#     lables = np.array(lables)
#     #条件判断，lables ='A'的下标索引
#     index_a = np.where(lables=='A')
#     index_b = np.where(lables=='B')
#     # print("A:",index_a)
#     # print("B:",index_b)
#     for i in lables:
#         if i == "A":
#             plt.scatter(group[index_a][:,:1],group[index_a][:,1:2],c="red")
#         elif i == 'B':
#             plt.scatter(group[index_b][:,:1],group[index_b][:,1:2],c="green")
#     plt.show()
#
# show_data(group,lables)
#
import numpy as np
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt

def KNN_classify(k,X_train,y_train,x):
  """
    k:表示knn的中k的值
    X_train: 训练集的features
    y_train: 训练集的labels
    x: 新的数据
  """
  assert 1<=k<=X_train.shape[0],"k must be valid"
  assert X_train.shape[0] == y_train.shape[0], \
  "the size of X_train must equal to the size of y_train"
  assert X_train.shape[1] == x.shape[0], \
  "the feature number of x must to be equal to X_train"
  # 计算新来的数据x与整个训练数据中每个样本数据的距离
  distances = [sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
  print(distances)
  nearest = np.argsort(distances) # 对距离排序并返回对应的索引
  print(nearest)

  topK_y = [y_train[i] for i in nearest] # 返回最近的k个距离对应的分类
  votes = Counter(topK_y) # 统计属于每个分类的样本数

  return votes.most_common(1)[0][0] # 返回属于样本数最多的分类结果

x = [[0,0],
     [10,12],
     [2,4],
     [6,5],
     [11,11],
     [12,16]]
y = [0,0,1,1,1,0]

X_train = np.array(x)
y_labels = np.array(y)

point = np.array([13,13])
type = (KNN_classify(2,X_train,y_labels,x))
 
if type == 1:
    color = 'red'
else:
    color = 'blue'
plt.scatter(x[0],x[1],color=color)

plt.show()