# HW03

> 用pytorch实现卷积神经网络，对cifar10数据集进行分类


# 网络结构

首先简单搭建了三层神经网络(Conv $\to$ ReLU $\to$ MaxPooling)，无 `batch_norm` 和正则化；获得基准结果，为`60%`左右
(以及微调`lr`，多次试验，验证基准线的信度)

不再头铁，考虑`batch_norm`，VGG16等双层`Conv`为一个块；提升到`70%`

对`classifier`进行改造，从简单的$512 \to 10$ 修改为 $512 \to 1025 \to 10$；期间尝试了中间层为$2048, 4096$，最终的`test_acc`无明显上升，但是到达`80%`所需的`epoch`变少，过拟合的更快

尝试不同`lr`, 发现1e-3对多种网络都很优，改变反而效果变差。

添加L2正则化，准确率曲线更加平滑

(期间：添加`early stop`，patience为10。在10次test_acc无提高后停止训练)

对Conv层的变化层次进行调整，发现扩大`channel`对于model的提升较大，此时达到`85%`

改最后一层为平均池化，效果不佳



# 数据增强

算`testData`, `trainData`的均值方差，进行归一化

- 对`test`仅仅进行归一化操作

- 对于`train`尝试如下
    - 仅仅左右翻转，上升至`83%`
    - 左右翻转 $+$ 高斯噪音， 上升至`84%`
    - 左右翻转 $+$ 饱和度等色彩的变化(分类结果小幅度改变色彩影响不大)，提升不大，但是训练时间大幅增加
    - 左右翻转 $+$ 仿射变换，训练太慢
    - 左右翻转 $+$ 色彩变化 $+$ 高斯噪音 $+$ 仿射变换, 训练不起来，一直上不了`80%`
    - 左右翻转 $+$ 随机擦除， 效果提升，速度影响不大, 效果有所提升

实行数据增强后，earlystop的时间得以延后，train与test的准确度差值显著缩小（添加early stop大约差值在'4%'， 100epoch也能较好地维持到`8%`以内）

```python
# 自定义高斯噪声变换
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, p={2})'.format(self.mean, self.std, self.p)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 归一化
])

# 数据增强的数据预处理方式
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # 随机缩放裁剪
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色扰动
    # transforms.RandomRotation(15),  # 小幅旋转
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # 随机擦除
    # AddGaussianNoise(mean=0., std=0.1, p=0.5),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
])
```

# 其余修改

为方便模块化以及显示图片，在jupyter-lab中进行了全部实验，同时修改训练流程的输出（tqdm进度条与图像绘制）

train_acc曲线锐度大，部分是因为只记录了小部分train_acc， 方便和test_acc一同绘制）

train_log文件夹中保存了部分训练结果（部分train因为感觉个人感觉过慢和过度overfit提前终止，没有进行自动的绘图和保存）



# 最终结果

TODO


