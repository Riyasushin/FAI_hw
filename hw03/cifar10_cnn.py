# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import datetime

# 定义超参数
batch_size = 64
learning_rate = 0.0001
num_epochs = 100

DEBUG = False



# for log
total_log = ""

# 用于 early stopping
best_accuracy = 0.0
counter = 0
patience = 5

# 定义数据预处理方式
# 普通的数据预处理方式
transform = transforms.Compose([
    transforms.ToTensor(),])
# 数据增强的数据预处理方式
# transform = transforms.Compose(



# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# images, labels = next(iter(train_loader))
# print(images.shape) # 64, 3, 32, 32

# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    def __init__(self, height, width):
        super(Net, self).__init__()

        self.image_height = height
        self.image_width = width

        # 先写死，后面改成根据输入参数构建网络维度的 TODO
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), # 3, H, W -> 32, H, W
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 32, H/2, W/2 -> 64, H/2, W/2
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, stride=1, padding=1), # 64, H/4, W/4 -> 128, H/4, W/4
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 128, H/4, W/4 -> 128, H/8, W/8
        )

        self.fc = nn.Sequential(
            nn.Linear(24 * (height // 8) * (width // 8), 512), # 128, H/4, W/4 -> 512
            nn.ReLU(inplace=True), 
            nn.Linear(512, 10), # 512 -> 10
            nn.Softmax(dim=1) # 10 -> 10
        )
    
    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: tensor, shape [batch_size, 3, H, W]
        '''
        res = self.conv(x)
        if DEBUG:
            print("conv output shape: ", res.shape)
        res = res.view(res.size(0), -1)
        if DEBUG:
            print("conv flatten output shape: ", res.shape)
        res = self.fc(res)
        return res

# ResNet
class BasicBlockForResNet(nn.Module):
    expansion = 1 # in_channel = expansion * out_channel
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockForResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # !!!!!! TODO important
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) 

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_channels != self.expansion * out_channels: # TODO
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,  self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True) # inplace=True, 直接在原来的内存上进行操作，节省内存开销

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out)) 
        out += self.shortcut(x) # 残差连接
        out = self.relu(out)
        return out

class ResNetTest(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetTest, self).__init__()


        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        # Kaiming初始化 # TODO 初始化，来自PPT    
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 构建多个残差块
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.layer4(out)  
        out = self.avg_pool(out) 
        out = out.view(out.size(0), -1) 
        out = self.dropout(out)
        out = self.fc(out)
        return out

# 实例化模型
# model = Net()
model = ResNetTest(BasicBlockForResNet, [2, 2, 2, 2])

use_mlu = False # 爱了学长，喜欢这个判断
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'MLU is not available, use {device} instead.')
    

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) ## TODO


# 训练模型
for epoch in range(num_epochs):

    # for log
    epoch_log = f"**Epoch {epoch + 1}/{num_epochs}:**\n"

    # 训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            train_log = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'\
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100)
            print(train_log)
            # total_log += train_log + "\n\n" # 不需要
    # epoch train_acc
    epoch_log += '- train_acc: {:.2f}%'.format(accuracy.item() * 100) + '    train_loss: {:.4f}\n'.format(loss.item())

    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        test_log = 'Test Accuracy of the model on the 10000 test images: {} %'.format(test_acc)
        print(test_log)
        # total_log += ('**' + test_log + '**' + "\n\n" + '\n')

        # epoch test_acc
        epoch_log += '- test_acc:   {:>4.2f}%'.format(test_acc) + '    test_loss:   {:>4.4f}\n'.format(loss.item())
    epoch_log += '\n'
    # if DEBUG:
    print(epoch_log)
    total_log += epoch_log + '\n'

    # early stopping
    if (test_acc > best_accuracy):
        best_accuracy = test_acc
        counter = 0
    else:
        counter += 1
        print(f"Early stopping counter: {counter}")
        if counter > patience:
            print("Early stopping")
            break

# 训练完成后请求用户输入消息
message = input("请输入消息: ")

# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 将信息写入log.md文件
with open('log.md', 'a') as f:
    f.write(f"## {current_time}\n")
    f.write(f"**Message:**\n     {message}\n\n")
    f.write(f'**Hyperparameters:**\
                \n- batch_size={batch_size},\
                \n- learning_rate={learning_rate},\
                \n- num_epochs={num_epochs},\
                \n- device={device}\
                \n- loss_function={str(criterion)}\
                \n- optim=Adam \n\n')
    f.write(f"**Model:**\n     {model}\n\n")
    f.write(f"{total_log}\n\n")

print("训练日志已写入log.md文件")