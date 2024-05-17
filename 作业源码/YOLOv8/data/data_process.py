import os
import cv2
import json


def process(dict1, shape):
    l = ''
    for i in range(len(dict1['left'])):
        l += str(dict1['label'][i]) + ' ' + \
             str((dict1['left'][i] + dict1['width'][i] / 2) / shape[1]) + ' ' + \
             str((dict1['top'][i] + dict1['height'][i] / 2) / shape[0]) + ' ' + \
             str(dict1['width'][i] / shape[1]) + ' ' + \
             str(dict1['height'][i] / shape[0]) \
             + '\n'
    return l


f = open(
    "train.json",
    encoding='utf-8')
print(f)
data = json.load(f)

# 用于构建标签，以构建训练YOLO的数据格式
for i in data:
    print(i)
    img = cv2.imread(r'train/images' + '/' + i)
    print(img)
    shape = img.shape
    f = open(r'train/labels/' + i[0:6] + '.txt', 'w')
    f.write(process(data[i], shape))
    f.close()
