import torch
from config import device
from config import config
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


def predict(model):
    class ImageDataset(Dataset):
        def __init__(self, folder_path, transform):
            self.transform = transform
            self.folder_path = folder_path
            self.image_list = os.listdir(folder_path)

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, index):
            image_name = self.image_list[index]
            image_path = os.path.join(self.folder_path, image_name)
            image = Image.open(image_path)
            image = self.transform(image)
            label = image_name
            return image, label

    transform = transforms.Compose([
        transforms.Resize((60, 120)),
        # transforms.RandomCrop((60, 60)),
        # transforms.ColorJitter(0.3, 0.3, 0.2),
        # transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(config['test_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model = model().to(device)
    model.load_state_dict(torch.load(config['save_path']))
    model.eval()
    test_pred = []
    for input, name in dataloader:
        if device == 'cuda':
            input = input.cuda()
        c0, c1, c2, c3, c4 = model(input)
        output = np.concatenate([
            c0.data.cpu().numpy(),
            c1.data.cpu().numpy(),
            c2.data.cpu().numpy(),
            c3.data.cpu().numpy(),
            c4.data.cpu().numpy()], axis=1)
        test_pred.append(output)

    test_pred = np.vstack(test_pred)
    print(test_pred[0])
    print(test_pred.shape)

    test_predict_label = np.vstack([
        test_pred[:, :11].argmax(1),
        test_pred[:, 11:22].argmax(1),
        test_pred[:, 22:33].argmax(1),
        test_pred[:, 33:44].argmax(1),
        test_pred[:, 44:55].argmax(1),
    ]).T
    print(test_predict_label)
    test_label_pred = []
    for x in test_predict_label:
        test_label_pred.append(''.join(map(str, x[x != 10])))
    print(test_label_pred)
    # 初始化一个空列表来存储处理后的结果
    processed_predictions = []

    # 遍历预测结果列表中的每个数组
    for prediction_array in test_label_pred:
        # 将数组转换成字符串
        prediction_string = ''.join(map(str, prediction_array))

        # 将中间的零替换成减号
        prediction_string = prediction_string.replace('0', '-')

        # 去除末尾的零
        processed_prediction = prediction_string.rstrip('-')

        # 将减号替换回零
        processed_prediction = processed_prediction.replace('-', '0')

        # 将结果添加到处理后的预测结果列表中
        processed_predictions.append(processed_prediction)

    # 输出处理后的结果
    print(processed_predictions)

    result_list = []
    for number_str in processed_predictions:
        if number_str:
            # number_without_zeros = number_str.replace('0', '')
            # result_list.append(int(number_without_zeros))
            number = int(number_str)
        else:
            number = 0
        result_list.append(number)

    print(result_list)

    df_submit = pd.read_csv(config['sample_path'])
    df_submit['file_code'] = result_list
    df_submit.to_csv(config['submit_name'], index=None)


if __name__ == '__main__':
    from network import svhn

    predict(svhn)
