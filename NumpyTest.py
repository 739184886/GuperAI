import numpy as np

# print(np.__version__)
# #生成一位数组
# data = [1,2,3,4,5,6]
# print(type(data),data)
# x = np.array(data)
# print(type(x),x)
# print(x.dtype,x.shape,x.ndim)

# #生成矩阵（二位数组）
# # data2 = [[1,2],[3,4],[5,]]
# data2 = [[1,2],[3,4],[5,3]]
# x2 = np.array(data2)
# print(x2.ndim)  #维度
# print(x2.shape)
# print(x2.size)

# #0矩阵 1矩阵
# x3 = np.zeros((2,3)) #2 第一维度包含两个元素 3第二维包含三个元素
# print(x3)
# print(x3.ndim,x3.shape)
# x4 = np.ones((2,3),dtype='int8')
# print(x4)

# #生成联系的元素
# print(np.arange(6))
# print(np.arange(1,8,2)) #1开始 到8   步长为2
#
# #使用astype复制数组，转换类型
# x = np.array([1,2,3,4,5],dtype=np.float32)
# y = x.astype(dtype=np.int8)
# print(x,y)
#
# d = y.astype(x.dtype)
# print(d)

# #数组与标量、数组预算
# x = np.array([1,2,3])
# print(x*2)
# print(x>2)
# y = np.array([2,3,4])
# print(x*y)#两个数组运算要shape一样

# #ndarray的索引， 降维
# x = np.array([[1,2],[3,4],[5,6]])
# print(x.shape)
# print(x[1][0])#==print(x[1,0])

# x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(x.shape)
# print(x[1,1,0])
#
# y = x[0].copy()#生成一个副本，x变化y不变
# print(y)
# z = x[0]#引用，x变，z就变
# print(z)

# #ndarray的切片, 不会改变维度
# x = np.arange(6)
# print(x)
# print(x[1:4])
# print(x[:3])
# print(x[0:4:2])#2步长
#
# x = np.array([[1,2],[3,4],[5,6]])
# print(x[:2])
# print(x[:2][:1])
# print(x[:2,:1])
#
# x = np.arange(24)
# x2 = x.reshape(12,2) #遍二维
# print(x2)

# # 花式索引
# x = np.array([1,2,3,4,5,6])
# print(x[[0,2]])
# print(x[[0,-2]])
# x = np.array([[1,2],[3,4],[5,6]])
# print(x[2,1])  #6
# print(x[[2,1]]) # [[5 6] [3 4]]
# print(x[[2,1],[0,1]]) # [5 4]
# print(x[0:2][0,0])#1
# print(x[0:2])

# #ndarray 数组转置和轴对称
# k = np.arange(12).reshape(3,4)
# print(k)
# #转置
# print(k.T)
# k = k.transpose(1,0)#轴变换 == k.T
# print(k)


# #高维数组的轴对象
# k = np.arange(24).reshape(2,3,4)#个数代表维度， 2.3.4每个维度的长度   此处轴可以理解为三层数组
# print(k)
# k = k.transpose(1,0,2)#指定轴变换 此处轴都要写上
# print(k)
# k = k.swapaxes(1,0)#指定的轴变换
# print(k)

# #基本统计
# x = np.arange(12).reshape(3,4)
# print(x)
# print(x.shape)
# print(x.mean())#均值
# print(x.mean(axis=0))#维度均值
# print(x.sum())
# print(x.sum(axis=1))#维度求和

# #ndarray的存取
# x = np.arange(8).reshape(4,2)
# np.save("file",x)#二进制数据存放，npy结尾
# y = np.load("file.npy")
# print(y)

# #矩阵求逆矩阵，不是所有矩阵都有逆矩阵
# x = np.array([[1,1],[1,2]])
# y = np.linalg.inv(x)
# print(y)

# #随机数
# x = np.random.randint(0,2,10000)#抛硬币概率 2是范围，10000是取多少次 也就是取 0 1随机10000
# print((x>0).sum())

# #where函数 重要的重要的
# cond = np.array([True,False,True,False,True])
# x = np.where(cond,1,-1)#三元运算符
# print(x)
# conArray = np.arange(4)
# print(conArray)
# w = np.where(conArray<3,1,-1)
# print(w)