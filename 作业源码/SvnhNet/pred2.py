import torch
from config import device
from config import config
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


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
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
        image = self.transform(image)
        label = image_name
        return image, label


def predict(model, config, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    # Uncomment the following lines if you want to use more advanced augmentations
    # transform = transforms.Compose([
    #     transforms.Resize((68, 136)),
    #     transforms.RandomCrop((64, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    dataset = ImageDataset(config['test_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model = model().to(device)
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    model.eval()

    test_pred = []
    for input, name in dataloader:
        input = input.to(device)
        with torch.no_grad():
            c0, c1, c2, c3, c4 = model(input)
            output = np.concatenate([
                c0.cpu().numpy(),
                c1.cpu().numpy(),
                c2.cpu().numpy(),
                c3.cpu().numpy(),
                c4.cpu().numpy()], axis=1)
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

    result_list = []
    for number in test_label_pred:
        number_str = str(number)
        result_list.append(int(number_str))

    print(len(test_label_pred))
    df_submit = pd.read_csv(config['sample_path'])
    df_submit['file_code'] = test_label_pred
    df_submit.to_csv(config['submit_name'], index=None)
