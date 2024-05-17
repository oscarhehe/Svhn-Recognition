from ultralytics import YOLO
import multiprocessing
import os
import csv

multiprocessing.freeze_support()


def extract_targets_from_txt(image_directory, txt_directory, csv_file):
    with open(csv_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['file_name', 'file_code'])

        for filename in os.listdir(image_directory):
            if filename.endswith('.png'):
                img_name = os.path.splitext(filename)[0]  # 获取图片名
                txt_path = os.path.join(txt_directory, f'{img_name}.txt')  # 对应的文本文件路径
                # 如果文本文件存在，读取标签内容
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as txtfile:
                        targets = []

                        # 读取每一行的目标类别
                        for line in txtfile:
                            line = line.strip()
                            if line:
                                target_class = line[0]
                                targets.append(target_class)

                        # 将目标类别列表连接成一个数字
                        file_code = ''.join(targets)

                        # 获取当前txt文件的文件名
                        file_name = os.path.splitext(img_name)[0]

                        # 写入csv文件中的一行
                        writer.writerow([file_name, file_code])
                else:
                    # 如果文本文件不存在，将标签设置为-1
                    writer.writerow([img_name, '-1'])


if __name__ == "__main__":
    yolo_vn = (r'C:\Users\Administrator\Desktop\神经网络与深度学习课程报告\Street-View-Character-Recognition-main'
               r'\Street-View-Character-Recognition-main\YOLO\runs\detect\train9\weights\best.pt')
    train_data = r'./data/mydata.yaml'
    model_best = (r'C:\Users\Administrator\Desktop\神经网络与深度学习课程报告\Street-View-Character-Recognition-main'
                  r'\Street-View-Character-Recognition-main\YOLO\runs\detect\train11\weights\best.pt')
    test_data = (r'C:\Users\Administrator\Desktop\神经网络与深度学习课程报告\Street-View-Character-Recognition-main\Street-View'
                 r'-Character-Recognition-main\data\mchar_test_a')
    txt_folder = './runs/detect/predict/val'
    csv_file = 'file.csv'

    model = YOLO(yolo_vn)
    model.train(data=train_data, batch=8, epochs=10)

    # model = YOLO(model_best)
    # model.predict(source=test_data, save_txt=True)

    # extract_targets_from_txt(test_data, txt_folder, csv_file)
