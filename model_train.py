import copy
import time
import os  # <--- 新增：导入os模块，用于操作文件路径

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet18, Residual  # 确保 model.py 已在Colab中加载
import torch.nn as nn
import pandas as pd

# <--- 新增：从google.colab导入drive模块
from google.colab import drive

# --- 1. 挂载谷歌云端硬盘 ---

drive.mount('/content/drive')

# --- 2. 定义在云盘中的保存目录 ---
# 您可以自定义这个路径。'/content/drive/My Drive/' 是您云盘的根目录
SAVE_DIR = '/content/drive/My Drive/Colab_Results/ResNet18_2'
# 如果目录不存在，则自动创建
os.makedirs(SAVE_DIR, exist_ok=True)


def train_val_data_process():
    # <--- 修改：请确保您的数据集已经上传到Colab或云盘，并修改这里的路径
    # 示例路径，假设您将 'data' 文件夹上传到了Colab的根目录下
    ROOT_TRAIN = r'data/train'

    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155, 0.06216329, 0.05930814])
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    # 在Colab中，num_workers建议设置为2
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # 训练循环
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 验证循环
        model.eval()
        with torch.no_grad():  # <--- 新增：在验证循环中关闭梯度计算，节省资源
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # --- 3. 将最佳模型权重保存到云盘 ---
    model.load_state_dict(best_model_wts)
    # <--- 修改：将原来的本地路径改为使用前面定义的云盘路径
    save_path = os.path.join(SAVE_DIR, 'best_model.pth')
    torch.save(best_model_wts, save_path)
    print(f'最佳模型已保存至: {save_path}')  # <--- 新增：打印保存路径，方便确认

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all, })

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    # --- 4. 将结果图表保存到云盘 ---
    # <--- 新增：在显示图表前，先将其保存到文件中
    plot_path = os.path.join(SAVE_DIR, 'training_plot.png')
    plt.savefig(plot_path)
    print(f'训练过程图已保存至: {plot_path}')  # <--- 新增：打印保存路径

    plt.show()


if __name__ == '__main__':
    ResNet18 = ResNet18(Residual)
    train_data, val_data = train_val_data_process()
    train_process = train_model_process(ResNet18, train_data, val_data, num_epochs=50)

    # --- 5. 将训练历史数据(DataFrame)保存为CSV文件到云盘 ---
    # <--- 新增：这是一个好习惯，便于后续分析数据
    history_path = os.path.join(SAVE_DIR, 'training_history.csv')
    train_process.to_csv(history_path, index=False)
    print(f'训练历史数据已保存至: {history_path}')  # <--- 新增：打印保存路径

    matplot_acc_loss(train_process)