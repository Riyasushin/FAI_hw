 HW 02

> RiJoshin
> 2400013201

## 01
> 用numpy实现训练MLP网络识别手写数字MNIST数据集

1. ReLU (Epoch 10, Test acc: 95.40%,  Loss: 0.1663)

   ![ReLU](ReLU.png)

2. Tanh (Epoch 10, Test acc: 94.45%, Loss: 0.2019)

   ![Tanh](Tanh.png)

3. Sigmoid (Epoch 10, Test acc: 87.06%, Loss: 0.5153)

   ![Sigmoid](Sigmoid.png)

## 02

> 使用Pytorch训练MNIST数据集的MLP模型任务介绍



### Adam + Dropout

(Epoch: 10, Loss: 0.0687, Acc: 98.30%)

![def_print](def_print.png)

![default](default.png)



| 部分网络结构                                    | Train_Acc(100%) | Val_Acc |
| ----------------------------------------------- | --------------- | ------- |
| 784-Dropout(.5)-800-Dropout(.5)-800-10 + Adam   | 98.12           | 98.30   |
| 784-800-800-10 + Adam                           | 99.44           | 98.06   |
| 784-Dropout(.5)-1000-Dropout(.5)-800-10 + Adam  | 99.40           | 98.16   |
| 784-Dropout(.1)-1000-Dropout(.5)-800-10 + Adam  | 99.49           | 97.72   |
| 784-Dropout(.9)-1000-Dropout(.5)-800-10 + Adam  | 99.48           | 97.80   |
| 784-1000-800-10 + Adam                          | 99.35           | 97.79   |
| 784-800-800-10 + SGD                            | 82.05           | 84.66   |
| 784-Dropout(.5)-900-Dropout(.5)-800-10-RMSprop- | 97.52           | 97.44   |
| 784-900-800-10-RMSprop                          | 99.51           | 97.54   |

### 修改的描述

- 各种显示操作的简化：plt保存，训练时候的打印数据
- 调参数
