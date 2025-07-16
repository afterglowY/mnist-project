import torch
print(torch.__version__)

'''
 MNIST包含70000张手写数字图像：60000用于训练，10000用于测试
 图像是灰度的，28×28像素的，并且居中的，以减少预处理和加快运行
'''
import torch
from torch import nn    #导入神经网络模块
from torch.utils.data import DataLoader  #数据包管理工具，打包数据
from torchvision import  datasets  #封装了很多与图像相关的模型，数据集
from torchvision.transforms import ToTensor  #数据转换，张量，将其他类型的数据转换为tensor张量，numpy array

'''下载训练数据集（包含训练图片+标签）'''
training_data = datasets.MNIST( #跳转到函数的内部源代码，pycharm按下ctrl + 鼠标点击
    root="data", #表示下载的手写数字  到哪个路径。60000
    train=True, #读取下载后的数据中的训练集
    download=True, #如果你之前已经下载过了，就不用下载
    transform=ToTensor(), #张量，图片是不能直接传入神经网络模型
 )   #对于pytorch库能够识别的数据一般是tensor张量


'''下载测试数据集（包含训练图片+标签）'''
test_data = datasets.MNIST( #跳转到函数的内部源代码，pycharm按下ctrl + 鼠标点击
    root="data", #表示下载的手写数字  到哪个路径。60000
    train=False, #读取下载后的数据中的训练集
    download=True, #如果你之前已经下载过了，就不用下载
    transform=ToTensor(), #Tensor是在深度学习中提出并广泛应用的数据类型
 )   #Numpy数组只能在CPU上运行。Tensor可以在GPU上运行。这在深度学习应用中可以显著提高计算速度。
print(len(training_data))

'''展示手写数字图片，把训练集中的59000张图片展示'''
from matplotlib import pyplot as plt
figure = plt.figure()
for i in range(9):
    img,label = training_data[i+59000] #提取第59000张图片

    figure.add_subplot(3,3,i+1) #图像窗口中创建多个小窗口，小窗口用于显示图片
    plt.title(label)
    plt.axis("off")  #plt.show(I) 显示矢量
    plt.imshow(img.squeeze(),cmap="gray") #plt.imshow()将Numpy数组data中的数据显示为图像，并在图形窗口中显示
    a = img.squeeze()  #img.squeeze()从张量img中去掉维度为1的，如果该维度的大小不为1，则张量不会改变
plt.show()

'''创建数据DataLoader（数据加载器）'''
# batch_size:将数据集分为多份，每一份为batch_size个数据
#       优点：可以减少内存的使用，提高训练速度

train_dataloader = DataLoader(training_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


'''判断当前设备是否支持GPU，其中mps是苹果m系列芯片的GPU'''
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")   #字符串的格式化，CUDA驱动软件的功能：pytorch能够去执行cuda的命令
# 神经网络的模型也需要传入到GPU，1个batch_size的数据集也需要传入到GPU，才可以进行训练


''' 定义神经网络  类的继承这种方式'''
class NeuralNetwork(nn.Module): #通过调用类的形式来使用神经网络，神经网络的模型，nn.mdoule
    def __init__(self): #python基础关于类，self类自己本身
        super().__init__() #继承的父类初始化
        self.flatten = nn.Flatten() #展开,创建一个展开对象flatten
        self.hidden1 = nn.Linear(28*28,128) #第1个参数：有多少个神经元传入进来，第2个参数：有多少个数据传出去
        self.hidden2 = nn.Linear(128,256) #第1个参数：有多少个神经元传入进来，第2个参数：有多少个数据传出去
        self.out = nn.Linear(256,10) #输出必须和标签的类别相同，输入必须是上一层的神经元个数
    def forward(self,x):   #前向传播，你得告诉它 数据的流向 是神经网络层连接起来，函数名称不能改
        x = self.flatten(x)  #图像进行展开
        x = self.hidden1(x)
        x = torch.relu(x)   #激活函数，torch使用的relu函数
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.out(x)
        return x
model = NeuralNetwork().to(device) #把刚刚创建的模型传入到GPU
print(model)

def train(dataloader,model,loss_fn,optimizer):
    model.train() #告诉模型，我要开始训练，模型中w进行随机化操作，已经更新w，在训练过程中，w会被修改的
# pytorch提供2种方式来切换训练和测试的模式，分别是：model.train() 和 mdoel.eval()
# 一般用法是：在训练开始之前写上model.train(),在测试时写上model.eval()
    batch_size_num = 1
    for X,y in dataloader:              #其中batch为每一个数据的编号
        X,y = X.to(device),y.to(device) #把训练数据集和标签传入cpu或GPU
        pred = model.forward(X)         # .forward可以被省略，父类种已经对此功能进行了设置
        loss = loss_fn(pred,y)          # 通过交叉熵损失函数计算损失值loss
        # Backpropagation 进来一个batch的数据，计算一次梯度，更新一次网络
        optimizer.zero_grad()           # 梯度值清零
        loss.backward()                 # 反向传播计算得到每个参数的梯度值w
        optimizer.step()                # 根据梯度更新网络w参数

        loss_value = loss.item()        # 从tensor数据种提取数据出来，tensor获取损失值
        if batch_size_num %100 ==0:
            print(f"loss: {loss_value:>7f} [number:{batch_size_num}]")
        batch_size_num += 1

def Test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)  #10000
    num_batches = len(dataloader)  # 打包的数量
    model.eval()        #测试，w就不能再更新
    test_loss,correct =0,0
    with torch.no_grad():       #一个上下文管理器，关闭梯度计算。当你确认不会调用Tensor.backward()的时候
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model.forward(X)
            test_loss += loss_fn(pred,y).item() #test_loss是会自动累加每一个批次的损失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            a = (pred.argmax(1) == y) #dim=1表示每一行中的最大值对应的索引号，dim=0表示每一列中的最大值对应的索引号
            b = (pred.argmax(1) == y).type(torch.float)
    test_loss /= num_batches #能来衡量模型测试的好坏
    correct /= size  #平均的正确率
    print(f"Test result: \n Accuracy:{(100*correct)}%, Avg loss:{test_loss}")

loss_fn = nn.CrossEntropyLoss()  #创建交叉熵损失函数对象，因为手写字识别一共有十种数字，输出会有10个结果
#
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) #创建一个优化器，SGD为随机梯度下降算法
# # params：要训练的参数，一般我们传入的都是model.parameters()
# # lr:learning_rate学习率，也就是步长
#
# # loss表示模型训练后的输出结果与样本标签的差距。如果差距越小，就表示模型训练越好，越逼近真实的模型
train(train_dataloader,model,loss_fn,optimizer) #训练1次完整的数据。多轮训练
Test(test_dataloader,model,loss_fn)

epochs = 10
for t in range(epochs):
    print(f"epoch {t+1}\n---------------")
    train(train_dataloader,model,loss_fn,optimizer)
print("Done!")
Test(test_dataloader,model,loss_fn)