# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
#
# batch_size = 100
# lr = 0.01
# num_epoch = 1
#
# # 数据归一化、标准化
# data_form = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
#
# # 从torchvision获取数据集
# train_dataset = datasets.MNIST(root="./MNIST_data", train=True, transform=data_form, download=True)
# test_dataset = datasets.MNIST(root="./MNIST_data", train=True, transform=data_form, download=False)
#
# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
#
# # print(train_dataset.train_data.size())
# # print(train_dataset.train_labels.size())
#
# class Net(nn.Module):
#     def __init__(self, in_dim, n_hidden1, n_hidden2, out_dim):
#         super(Net, self).__init__()
#         # 初始输入、隐藏、输出层
#         # Sequential 按顺序执行，先liner然后relue;   true  liner的输出不保留，直接被relu计算覆盖
#         self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.ReLU(inplace=True))
#         self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.ReLU(inplace=True))
#         self.layer3 = nn.Sequential(nn.Linear(n_hidden2, out_dim))
#
#     def forward(self, x):
#         y1 = self.layer1(x)
#         y2 = self.layer2(y1)
#         y3 = self.layer3(y2)
#
#         return y3
#
# net = Net(784, 256, 128, 10)
#
# if torch.cuda.is_available():
#     net = net.cuda()
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#
# ##标记训练数据
# net.train()
# for epoch in range(num_epoch):
#     for i, (img, label) in enumerate(train_loader):
#         # print(img.size())
#         img = img.reshape(img.size(0), -1)  # 形状转换
#
#     if torch.cuda.is_available():
#         img = img.cuda()
#         label = label.cuda()
#
#     out = net(img)
#
#     loss_ = loss_fn(out, label)  # [100,10]
#     optimizer.zero_grad()
#     loss_.backward()
#     optimizer.step()
#
#     if i % 10 == 0:
#         print("epoch:{},i:{},loss{:,3}".format(epoch, i, loss_.item()))  # loss{:,3}三位精度
#
# #评估模型
# net.eval()
# eval_loss = 0.0#所有数据损失
# evl_acc = 0
# for date in test_loader:
#     img,label = date
#     img = img.reshape(img.size(0),-1)
#     if torch.cuda.is_available():
#         img = img.cuda()
#         label = label.cuda()# [100,1]
#     out = net(img)
#     loss_ = loss_fn(out, label)  # [100,10]
#
#     eval_loss += loss_.item() * label.size()
#     #计算精度
#     max_out = torch.argmax(out,1)#取最大值的索引 1代表轴
#
#     acc = (label == max_out).sum()
#     evl_acc += acc.item()
#
# print(label)
# print(torch.argmax(out,1))
# print("test_loss:{:.3},test_acc:{:.3}".format(
#     eval_loss / (len(test_dataset)),
#     evl_acc / (len(test_dataset))
# ))
#
# '''老师的代码'''
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

batch_size = 100
learning_rate = 0.01
num_epoches = 5

# 数据预处理：
# transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0], [1])])

train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=data_tf,
                               download=True)  # 从torchvision包中下载数据集，并且保存，指明属于训练或测试数据,转换成Tensor()，一次性下载所有数据。
test_dataset = datasets.MNIST(root='./MNIST_data', train=False, transform=data_tf,
                              download=False)  # 测试集的数据不需要再单独下载，已经在第一次统一下载了

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)  # 数据加载器，获取训练数据，从下载的数据中获取数据，并且选择获取批次、每次获取是否打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 数据加载器，获取测试数据

# print(train_dataset.train_data.size())
# print(train_dataset.train_labels.size())
#
# print(test_dataset.test_data.size())
# print(test_dataset.test_labels.size())


class Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        # Sequential()函数的功能是将网络的层组合到一起,按顺序执行

        # 例如torch.nn.ReLU(inplace=True)
        # inplace=True表示进行原地操作，对上一层传递下来的tensor直接进行修改，如x=x+3；
        # inplace=False表示新建一个变量存储操作结果，如y=x+3，x=y；
        # inplace=True可以节省运算内存，不用多存储变量。
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1, bias=True), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2, bias=True), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim, bias=True))

    def forward(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)

        return y3


net = Net(28 * 28, 256, 128, 10)

if torch.cuda.is_available():
    net = net.cuda()

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()#交叉熵
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

net.train()
for epoch in range(num_epoches):
    for i, (img, label) in enumerate(train_loader):
        # print(img.size())
        #  img = img.view(img.size(0), -1)
        # img = img.reshape(-1, 784)#转换形状
        img = img.reshape(img.size(0), -1)  # 转换形状
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        out = net(img)
        loss = loss_fn(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('epoch: {},i: {}, loss: {:.3}'.format(epoch, i, loss.data.item()))  # 损失值显示3位精度
    # print('epoch: {}, loss: {:.3}'.format(epoch, loss.data.item()))  # 损失值显示3位精度

# 模型评估,此模式下，会固定模型中的BN层和Drpout层。
net.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    # img = img.reshape(-1,784)
    img = img.reshape(img.size(0), -1)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    out = net(img)
    loss = loss_fn(out, label)
    eval_loss += loss.data.item() * label.size(0)  # 平均损失*批次=每批数据的损失，  每批数据的损失*循环次数（+=叠加）=测试数据集的总损失
    pred = torch.argmax(out, 1)  # 返回每行中的最大值和最大值在每行中的索引

    num_correct = (pred == label).sum()  # 统计每批数据的精度
    eval_acc += num_correct.item()  # 每批的精度*循环次数（+=叠加）=测试数据集的总损失
'已经评估完所有测试集数据'

print(torch.argmax(out, 1))
print(label)
print(torch.max(out, 1))
print('Test Loss: {:.3}, Acc: {:.3}'.format(
    eval_loss / (len(test_dataset)),  # 计算所有测试数据集里的平均损失
    eval_acc / (len(test_dataset))  # 计算所有测试数据集里的平均精度
))