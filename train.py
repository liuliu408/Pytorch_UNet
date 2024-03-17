from model import UNet                  # 导入Unet 网络
from dataset import Data_Loader         # 数据处理
from torch import optim
import torch.nn as nn
import torch


# 网络训练模块
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # GPU or CPU
print(device)
net = UNet(in_channels=1, num_classes=1)                                # 加载网络
net.to(device)                                                          # 将网络加载到device上

# 加载训练集
train_path = "./data/train/image"
trainset = Data_Loader(train_path)
train_loader = torch.utils.data.DataLoader(dataset=trainset,batch_size=2,shuffle=True)

# len(trainset)  样本总数：21

# 加载测试集
test_path = "./data/test/image"
testset = Data_Loader(test_path)
test_loader = torch.utils.data.DataLoader(dataset=testset,batch_size=2)

optimizer = optim.RMSprop(net.parameters(),lr = 0.000001,weight_decay=1e-8,momentum=0.9)     # 定义优化器
criterion = nn.BCEWithLogitsLoss()                                                           # 定义损失函数

save_path = './UNet.pth'        # 网络参数的保存路径
best_acc = 0.0                  # 保存最好的准确率


for epoch in range(20):

    net.train()     # 训练模式
    running_loss = 0.0

    for image,label in train_loader:                   # 读取数据和label

        optimizer.zero_grad()                          # 梯度清零
        pred = net(image.to(device))                   # 前向传播

        loss = criterion(pred, label.to(device))       # 计算损失
        loss.backward()                                # 反向传播
        optimizer.step()                               # 梯度下降

        running_loss += loss.item()                    # 计算损失和

    net.eval()  # 测试模式
    acc = 0.0   # 正确率
    total = 0
    with torch.no_grad():
        for test_image, test_label in test_loader:

            outputs = net(test_image.to(device))     # 前向传播

            outputs[outputs >= 0] = 1  # 将预测图片转为二值图片
            outputs[outputs < 0] = 0

            acc += (outputs == test_label.to(device)).sum().item() / (480*480)     # 计算预测图片与真实图片像素点一致的精度：acc = 相同的 / 总个数
            total += test_label.size(0)

    accurate = acc / total  # 计算整个test上面的正确率
    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f %%' %
          (epoch + 1, running_loss, accurate*100))

    if accurate > best_acc:     # 保留最好的精度
        best_acc = accurate
        torch.save(net.state_dict(), save_path)     # 保存网络参数
