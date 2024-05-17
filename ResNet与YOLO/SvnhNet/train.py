import json
import math

import numpy as np
import torch
from config import device
import torch.nn as nn
from config import config
from utils import same_seed
from tqdm import tqdm
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def training(train_loader, valid_loader, model):
    same_seed(config['seed'])
    model = model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])
    # 切换模型为训练模式
    model.train()
    n_epochs = config['n_epochs']
    best_loss = math.inf
    for epoch in range(n_epochs):
        train_loss = []
        early_stop_count = 0
        train_pbar = tqdm(train_loader, leave=True, position=0)
        for input, target in train_pbar:
            optimizer.zero_grad()
            target = torch.tensor(target).type(torch.LongTensor)
            if device == 'cuda':
                input = input.cuda()
                target = target.cuda()
            c0, c1, c2, c3, c4 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3]) + \
                   criterion(c4, target[:, 4])
            # loss /= 5
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        train_loss_mean = sum(train_loss) / len(train_loss)
        # 切换模型为预测模型
        model.eval()
        val_loss = []
        # 不记录模型梯度信息
        with torch.no_grad():
            val_pred = []
            for input, target in valid_loader:
                target = torch.tensor(target).type(torch.LongTensor)
                if device == 'cuda':
                    input = input.cuda()
                    target = target.cuda()

                c0, c1, c2, c3, c4 = model(input)
                output = np.concatenate([
                    c0.data.cpu().numpy(),
                    c1.data.cpu().numpy(),
                    c2.data.cpu().numpy(),
                    c3.data.cpu().numpy(),
                    c4.data.cpu().numpy()], axis=1)
                val_pred.append(output)
                # 计算损失
                loss = criterion(c0, target[:, 0]) + \
                       criterion(c1, target[:, 1]) + \
                       criterion(c2, target[:, 2]) + \
                       criterion(c3, target[:, 3]) + \
                       criterion(c4, target[:, 4])
                # loss /= 5
                val_loss.append(loss.item())
        valid_loss_mean = sum(val_loss) / len(val_loss)
        # 计算在测试集上的准确率
        val_pred = np.vstack(val_pred)
        val_predict_label = np.vstack([
            val_pred[:, :11].argmax(1),
            val_pred[:, 11:22].argmax(1),
            val_pred[:, 22:33].argmax(1),
            val_pred[:, 33:44].argmax(1),
            val_pred[:, 44:55].argmax(1),
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        # 读取标签
        val_json = json.load(open('../data/new_mchar_val.json'))
        labels = [val_json[x]['label'] for x in val_json]
        labels = [''.join(map(str, x)) for x in labels]

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(labels))

        print(f'[{epoch + 1}/{n_epochs}]:train_loss:{round(train_loss_mean, 3)},valid_loss:{round(valid_loss_mean, 3)},'
              f'valid_acc:{round(val_char_acc, 3)}')

        # 保存最优模型以便之后加载使用
        if valid_loss_mean < best_loss:
            best_loss = valid_loss_mean
            torch.save(model.state_dict(), config['save_path'])
            print('saving models with loss {}'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1  # 引入早停机制

            # 在第十个epoch后降低学习率为原来的0.1
            if epoch >= 19:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
        if early_stop_count == config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
    return best_loss
