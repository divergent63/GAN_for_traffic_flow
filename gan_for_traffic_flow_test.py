import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pathlib import Path
import os, sys

import pandas as pd

from matplotlib import pyplot as plt
from random import choice

SAMPLE_GAP = 0.2
SAMPLE_NUM = 288
N_GNET = 50
BATCH_SIZE = 64
USE_CUDA = True
MAX_EPOCH = 10000


# 判别器
class disciminator(nn.Module):
    def __init__(self):
        super(disciminator, self).__init__()
        self.fc1 = nn.Linear(SAMPLE_NUM, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


# 生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(N_GNET, 128)
        self.fc2 = nn.Linear(128, SAMPLE_NUM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_data():
    df_pems_160104 = pd.read_csv(Path(os.getcwd()) / 'data' / 'PeMS' / 'd04_text_station_5min_2016_01_04.csv',
                                 header=None)
    df_pems_160104 = df_pems_160104.loc[df_pems_160104[1] == 400001]
    df_pems_160104 = pd.DataFrame(df_pems_160104).sort_values(by=0)

    df_pems_160104_fillna = df_pems_160104.fillna(0)

    POINT_pems = range(len(df_pems_160104_fillna.values))

    real_data_pems = df_pems_160104_fillna[9].values
    real_data_pems_400001 = real_data_pems
    # plt.plot(real_data_pems_400001)
    # plt.show()

    real_data_pems = (np.array(real_data_pems) - np.min(real_data_pems)) / (
                np.max(real_data_pems) - np.min(real_data_pems))
    return real_data_pems, POINT_pems


def MaxMinNormalization(x):
    x = (np.array(x) - np.min(x) / (np.max(x) - np.min(x)))
    return x


def main():
    real_data_pems, POINT_pems = load_data()

    # POINT_pems = MaxMinNormalization(POINT_pems)

    plt.ion()  # 开启interactive mode，便于连续plot
    # 用于计算的设备 CPU or GPU
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # 定义判别器与生成器的网络
    net_d = disciminator()
    net_g = generator()
    net_d.to(device)
    net_g.to(device)
    # 损失函数
    criterion = nn.BCELoss().to(device)
    # 真假数据的标签
    true_lable = Variable(torch.ones(BATCH_SIZE)).to(device)
    fake_lable = Variable(torch.zeros(BATCH_SIZE)).to(device)
    # 优化器
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.0001)
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=0.0001)

    losses_list = []
    for i in range(MAX_EPOCH):
        # 为真实数据加上噪声
        # print(i)
        # real_data = np.vstack([real_data_pems + np.random.normal(0, 0.01, SAMPLE_NUM) for _ in range(BATCH_SIZE)])
        real_data = np.vstack([real_data_pems + np.random.normal(0, 0.001, SAMPLE_NUM) for _ in range(BATCH_SIZE)])

        real_data = Variable(torch.Tensor(real_data)).to(device)
        # 用随机噪声作为生成器的输入
        g_noises = np.random.randn(BATCH_SIZE, N_GNET)
        g_noises = Variable(torch.Tensor(g_noises)).to(device)

        for _ in range(3):
            # 训练判别器
            optimizer_d.zero_grad()
            # 判别器判别真图的loss
            d_real = net_d(real_data)
            loss_d_real = criterion(d_real, true_lable)
            loss_d_real.backward()
            # 判别器判别假图的loss
            fake_date = net_g(g_noises)
            d_fake = net_d(fake_date)
            loss_d_fake = criterion(d_fake, fake_lable)
            loss_d_fake.backward()
            optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        fake_date = net_g(g_noises)
        d_fake = net_d(fake_date)
        # 生成器生成假图的loss
        loss_g = criterion(d_fake, true_lable)
        loss_g.backward()
        optimizer_g.step()

        prob = (loss_d_real.mean() + 1 - loss_d_fake.mean()) / 2.
        mse = np.mean((real_data[0].to('cpu').detach().numpy() - fake_date[0].to('cpu').detach().numpy()) ** 2)

        # 保存相关数据
        losses_list.append([prob.cpu().detach().numpy(), loss_d_real.cpu().detach().numpy(), loss_d_fake.cpu().detach().numpy(),
                            loss_g.cpu().detach().numpy(), real_data[0].to('cpu').detach().numpy(), fake_date[0].to('cpu').detach().numpy(),
                            mse])

        # 每200步画出生成的数字图片和相关的数据
        if i % 200 == 0:
            # print('fake_date[0]:  \n', fake_date[0])
            plt.cla()
            plt.plot(POINT_pems[0:2000], fake_date[0].to('cpu').detach().numpy()[0:2000], c='#4AD631', lw=2,
                     label="generated line")  # 生成网络生成的数据
            plt.plot(POINT_pems[0:2000], real_data[0].to('cpu').detach().numpy()[0:2000], c='#74BCFF', lw=3,
                     label="real flow")  # 真实数据
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % (prob),
                     fontdict={'size': 15})
            plt.ylim(-1, 1.5)
            plt.legend()
            plt.savefig(Path(os.getcwd()) / 'results' / 'traffic_flow' / Path(str(i) + '_epoch_plot.jpeg'), dpi=300)
            plt.draw(), plt.pause(0.2)

    # 保存相关数据
    losses_df = pd.DataFrame(losses_list, columns=['d_accuracy', 'd_real', 'd_fake', 'g', 'real', 'fake', 'mse'])
    losses_df_csv = losses_df
    losses_df_csv.to_csv("losses.csv")

    plt.ioff()
    plt.show()


if __name__ == '__main__':

    main()
