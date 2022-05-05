import cv2
import matplotlib.pyplot as plt
from imageai.Prediction.Custom import CustomImagePrediction
from logger import Logger
import os
import queue
import time
import threading
import json
import time

# get 當下目錄位置
execution_path = os.getcwd()

prediction_strip = []
prediction_piece = []


class TestPredictionThread:
    '''
    ImageAI Custom Prediction功能物件，針對鏡面汙染圖檔進行預測分類與報告產出
    '''

    def __init__(self, path):
        self.logger = Logger.__call__().get_logger()
        self.test_image_folder = self.get_all_folders(path)
        self.test_image_queue = self.get_all_images(path)
        self.lib_folder_path = path
        self.thread_num = 3

    # def get_all_images(self, path):
    #     images=[]
    #     for root, dirs, files in os.walk(path):
    #         for f in files:
    #             fullpath = os.path.join(root, f)
    #             images.append(fullpath)
    #     print("all image: ",len(images))
    #     return images

    def get_all_images(self, folder_path: str) -> queue:
        '''取得所有測試圖片

        '''
        image_queue = queue.Queue()
        for root, dirs, files in os.walk(
                os.path.join(execution_path, folder_path, "test")):
            for f in files:
                fullpath = os.path.join(root, f)
                image_queue.put(fullpath)
        self.logger.info(f"all image: {image_queue.qsize()}")
        # print("all image: ",image_queue.qsize())
        return image_queue

    def get_all_folders(self, folder_path: str) -> list:
        '''取得所有測試圖片的資料夾路徑

        Args:
            folder_path: 放test的檔案夾位置

        Returns:
            存放所有folder的名稱的list
        '''
        folder = []
        for root, dirs, files in os.walk(
                os.path.join(execution_path, folder_path, "test")):
            for f in dirs:
                fullpath = os.path.join(root, f)
                folder.append(fullpath)
        # print("all prediction folder: ", folder)
        return folder

    def get_predection_model(self, folder_path: str) -> str:
        '''取得訓練出來的最高信度的model

        Args:
            folder_path: 放model的檔案夾位置

        Returns:
            最高信度model的位置
        '''
        models_path = os.path.join(execution_path, folder_path, "models")
        all_models = os.listdir(models_path)
        all_prediction_value = [float(model[17:25]) for model in all_models]
        top_model_position = all_prediction_value.index(
            max(all_prediction_value))
        top_model_path = os.path.join(execution_path, folder_path, "models",
                                      all_models[top_model_position])
        return top_model_path

    def get_predection(self, folder_path: str, thread_num: str):
        '''取得Custom Prediction AI Image功能

        Args:
            folder_path: 欲使用之model的位置, thread_num: thread的編號
            thread_num: thread的編號

        Returns:
            custom prediction模組，裡面包含設定的 setModelType、setModelType、setJsonPath、loadModel
        '''
        prediction_model = self.get_predection_model(folder_path)
        prediction_json, model_class_info = self.get_model_class(folder_path)

        prediction = CustomImagePrediction()
        prediction.setModelTypeAsResNet()
        prediction.setModelPath(prediction_model)
        prediction.setJsonPath(prediction_json)
        prediction.loadModel(num_objects=len(model_class_info))
        self.logger.info(
            f"Thread[{thread_num}], Model path:{prediction_model}, Detect type: ResNet"
        )
        # print("Thread",thread_num, ", Model path: ",prediction_model, ", Detect type: ResNet")
        return prediction

    def get_model_class(self, folder_path: str) -> tuple:
        '''
        取得model class info，需要提供lib圖檔文件夾路徑
        '''
        json_path = os.path.join(execution_path, folder_path, "json")
        all_json_files = os.listdir(json_path)
        json_file = os.path.join(json_path, all_json_files[0])
        with open(json_file, 'r') as f:
            data = json.load(f)
            f.close()
        return json_file, data

    def image_prediction(self, thread_num: str):
        '''
        取得prediction image的結果,需要提供thread編號
        '''
        prediction = self.get_predection(self.lib_folder_path, thread_num)
        while self.test_image_queue.qsize() > 0:
            # 取得辨識圖檔的位置
            image_path = self.test_image_queue.get()
            # print("queue size: ",self.queue.qsize())

            # 處理辨識資料
            predictions, probabilities = prediction.predictImage(
                image_path, result_count=1)
            for eachPrediction, eachProbability in zip(predictions,
                                                       probabilities):
                print(
                    f"Thread[{thread_num}]  Image:{image_path} ====>  Prediction: {eachPrediction} : {eachProbability}"
                )
                if eachPrediction == "piece":
                    prediction_piece.append(image_path)
                if eachPrediction == "strip":
                    prediction_strip.append(image_path)
            print(
                '-------------------------------------------------------------------------------------------------------------------------'
            )

    def run(self):
        '''
        執行mutithreading跑custom prediction image
        '''
        thread_list = []

        for i in range(self.thread_num):
            process = threading.Thread(target=self.image_prediction,
                                       args=(str(i), ),
                                       daemon=True)
            thread_list.append(process)

        for i in range(self.thread_num):
            thread_list[i].start()

        for i in range(self.thread_num):
            thread_list[i].join()

        # for i in range(len(self.prediction_folder)):
        #     prediction_images = self.get_all_images(self.prediction_folder[i])
        #     process = threading.Thread(target=self.dosomething, args=(str(i), prediction_images, ), daemon=True)
        #     thread_list.append(process)


time_1 = time.time()
d = TestPredictionThread("idenprof")
d.run()
print("prediction piece: ", len(prediction_piece))
print("prediction strip: ", len(prediction_strip))
time_2 = time.time()
time_interval = time_2 - time_1
print("time: ", time_interval)
