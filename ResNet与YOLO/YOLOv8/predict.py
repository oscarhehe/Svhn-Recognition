from main import *
import os


def sort_prediction_files(directory):
    def read_predictions(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        predictions = [line.strip().split() for line in lines]
        return predictions

    def write_predictions(file_path, predictions):
        with open(file_path, 'w') as file:
            for prediction in predictions:
                file.write(' '.join(prediction) + '\n')

    def sort_predictions(predictions):
        # 将预测结果中的 x1 坐标转换为浮点数并按 x1 坐标排序
        predictions.sort(key=lambda x: float(x[1]))
        return predictions

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            predictions = read_predictions(file_path)
            sorted_predictions = sort_predictions(predictions)
            write_predictions(file_path, sorted_predictions)


test_data = (r'C:\Users\Administrator\Desktop\神经网络与深度学习课程报告\Street-View-Character-Recognition-main'
             r'\Street-View-Character-Recognition-main\YOLO\data\test\images')
txt_folder = './runs/detect/predict/labels'
csv_file = 'file3.csv'
model_best = (r'C:\Users\Administrator\Desktop\神经网络与深度学习课程报告\Street-View-Character-Recognition-main'
              r'\Street-View-Character-Recognition-main\YOLO\runs\detect\train9\weights\best.pt')

model = YOLO(model_best)
model.predict(source=test_data, save_txt=True)


sort_prediction_files(txt_folder)

extract_targets_from_txt(test_data, txt_folder, csv_file)
