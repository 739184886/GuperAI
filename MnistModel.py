import torch
from torch.utils.data import dataloader
from torchvision import datasets,transforms


databatch = 100
lr = 0.01

mnist_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0],[1])])

trainDataSet = datasets.MNIST(root='./MNIST_data',train=True,transform=mnist_transform,download=True)
testDataSet = datasets.MNIST(root='./MNIST_data',train=False,transform=mnist_transform,download=False)

trainDataLoader = dataloader.DataLoader(trainDataSet,batch_size=databatch,shuffle=True)
testDataLoader = dataloader.DataLoader(testDataSet,batch_size=databatch,shuffle=False)

# print(trainDataSet.train_data.size())
# print(trainDataSet.train_labels.size())

class MyMnistNet(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(MyMnistNet,self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_dim,n_hidden_1,bias=True),torch.nn.ReLU(inplace=True))
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(n_hidden_1,n_hidden_2,bias=True),torch.nn.ReLU(inplace=True))
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(n_hidden_2,out_dim,bias=True),torch.nn.ReLU(inplace=True))

    def forward(self,x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)

        return y3

net = MyMnistNet(28 * 28, 256, 128, 10)
if torch.cuda.is_available():
    net = net.cuda()

loss_fn = torch.nn.CrossEntropyLoss()#交叉熵
optimizer = torch.optim.SGD(net.parameters(),lr=lr)

for epoch in range(3):
    for i,(img, label) in enumerate(trainDataLoader):
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
            print('epoch: {},i: {}, loss: {:.3}'.format(epoch, i, print_loss))  # 损失值显示3位精度

