import cv2
import numpy as np
import time
import queue
import os
import threading
from logger import Logger


class ImageProcess:

    def __init__(self):
        self.HSV = None
        self.BGR = None
        self.GRAY = None
        self.mouse_click = None
        self._pollution_hsv_lower = (0, 80, 89)
        self._pollution_hsv_upper = (10, 255, 255)
        # self._frame_hsv_lower1 = (50, 216, 87)
        # self._frame_hsv_upper1 = (58, 255, 162)
        self._frame_hsv_lower = (51, 45, 51)
        self._frame_hsv_upper = (61, 255, 167)
        self._au_hsv_lower = (60, 255, 255)
        self._au_hsv_upper = (99, 255, 255)
        self._square_fit_size = 224
        self._folder_root = None
        self._folder_list = None
        self._image_queue = None
        self._thread_num = 3
        # cv2.WINDOW_KEEPRATIO->自適比例 cv2.WINDOW_NORMAL->可調視窗
        self._windows_size = cv2.WINDOW_NORMAL
        self._logger = Logger.__call__().get_logger()

    def _get_color_range(self, area: str) -> dict:
        '''取得HSVlower和upper的range

        Args:
            area: 取得的範圍可以選取"frame"或"pullution"

        Returns:
            dict: lower和upper的數據

        '''
        if isinstance(area, str):
            if area == 'frame':
                return {
                    'LowerbH': 35,
                    'LowerbS': 43,
                    'LowerbV': 46,
                    'UpperbH': 58,
                    'UpperbS': 255,
                    'UpperbV': 140
                }
            if area == 'pollution':
                return {
                    'LowerbH': 0,
                    'LowerbS': 43,
                    'LowerbV': 46,
                    'UpperbH': 10,
                    'UpperbS': 255,
                    'UpperbV': 255
                }
        else:
            raise ValueError('area value must be "frame" or "pollution"')

    def _cv_imread(self, file_path: str) -> object:
        '''將圖片資料透過numpy讀取二進制資料並將其解碼維圖像數據
           cv預設讀取格是為BGR

        Args:
            file_path: 圖片路徑

        Returns: 
            cv_img: 轉為cv的圖形資料格式

        '''
        if isinstance(file_path, str) or len(file_path) < 0:
            cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            return cv_img
        else:
            raise ValueError('Please check your file path!')

    def _image_resize(self, image: np.ndarray) -> np.ndarray:
        '''將圖片資料重新縮放

        Args:
            image: cv2的numpy格式資料

        Returns: 
            resize_image: 縮放後的圖片資料

        '''
        if isinstance(image, np.ndarray):
            height, width, channels = image.shape
            fit_size = self._square_fit_size
            if width > fit_size and height > fit_size:
                if width > height:
                    height = int((fit_size / width) * height)
                    width = fit_size
                else:
                    width = int((fit_size / height) * width)
                    height = fit_size
            resize_image = cv2.resize(image, (width, height),
                                      interpolation=cv2.INTER_AREA)

            return resize_image
        else:
            raise ValueError('Please check your image!')

    def _image_rotate(self, pollution: np.ndarray) -> np.ndarray:
        '''將圖片旋轉180度處理

        Args:
            pollution: cv2的numpy格式資料

        Returns: 
            image_rotate: 旋轉後的圖片資料格式

        '''
        if isinstance(pollution, np.ndarray):
            image_rotate = cv2.rotate(pollution, cv2.ROTATE_180)
            return image_rotate
        else:
            raise ValueError(
                'input error, Please check your pollution content!')

    def _image_flip(self, pollution: np.ndarray) -> np.ndarray:
        '''將圖片資料鏡像處理

        Args:
            pollution: cv2的numpy格式資料

        Returns: 
            image_flip: 轉為HSV資料格式

        '''
        if isinstance(pollution, np.ndarray):
            image_flip = cv2.flip(pollution, 1)
            return image_flip
        else:
            raise ValueError(
                'input error, Please check your pollution content!')

    def _image_merge(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        '''將圖片資料結合

        Args:
            mask1: 欲結合的cv2的numpy格式資料1
            mask2: 欲結合的cv2的numpy格式資料2

        Returns: 
            image_merge: 結合後的內容

        '''
        if isinstance(mask1, np.ndarray):
            image_merge = cv2.bitwise_xor(mask1, mask2)
            return image_merge
        else:
            raise ValueError(
                'input error, Please check your pollution content!')

    def _BGR_to_RGB(self, image: np.ndarray) -> np.ndarray:
        '''將圖片資料格式轉為RGB格式

        Args:
            image: cv2的numpy格式資料

        Returns: 
            image_rgb: 轉為RGB資料格式

        '''
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image_rgb
        else:
            raise ValueError('Please check your image content!')

    def _BGR_to_GRAY(self, image: np.ndarray) -> np.ndarray:
        '''將圖片資料轉為GRAY格式

        Args:
            image: cv2的numpy格式資料

        Returns: 
            image_gray: 轉為GRAY資料格式

        '''
        if isinstance(image, np.ndarray):
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image_gray
        else:
            raise ValueError('Please check your image content!')

    def _BGR_to_HSV(self, image: np.ndarray) -> np.ndarray:
        '''將圖片資料轉為HSV格式

        Args:
            image: cv2的numpy格式資料

        Returns: 
            image_hsv: 轉為HSV資料格式

        '''
        if isinstance(image, np.ndarray):
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return image_hsv
        else:
            raise ValueError('Please check your image!')

    def _HSV_to_BGR(self, image: np.ndarray) -> np.ndarray:
        '''將圖片資料從HSV轉為BGR格式

        Args:
            image: cv2的numpy格式資料

        Returns: 
            image_hsv2bgr: HSV轉為BGR資料格式

        '''
        if isinstance(image, np.ndarray):
            image_hsv2bgr = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return image_hsv2bgr
        else:
            raise ValueError('Please check your image!')

    def _mouse_click(self, event, x, y, flags, para):
        '''滑鼠動作取值

        Args:
            event: 滑鼠點擊事件
            x: 滑鼠X座標
            y: 滑鼠Y座標
            flags: 滑鼠拖曳事件
            para: 附帶參數

        Returns: None

        '''
        click_result = {}
        if event == cv2.EVENT_LBUTTONDOWN:
            click_result["BGR"] = self.BGR[y, x]
            click_result["GRAY"] = self.GRAY[y, x]
            click_result["HSV"] = self.HSV[y, x]
            print("BGR:", self.BGR[y, x])
            print("GRAY:", self.GRAY[y, x])
            print("HSV:", self.HSV[y, x])
            print("=" * 30)
            self.mouse_click = click_result

    def _frame_area(self, image: np.ndarray, color_type: str) -> np.ndarray:
        '''取出圖片frame部分

        Args:
            image: cv2 np array的資料格式
            color_type: image的color型態

        Returns: 
            frame_mask2: frame hsv的資料
        '''
        if isinstance(image, np.ndarray) and isinstance(color_type, str):
            # frame_mask1 = cv2.inRange(image, self._frame_lower1,
            #                           self._frame_upper1)
            # frame_mask2 = cv2.inRange(image, self._frame_hsv_lower2,
            #                           self._frame_hsv_upper2)
            # frame_area = cv2.bitwise_or(image, frame_mask2)
            image_copy = image.copy()
            image_hsv = self._BGR_to_HSV(image_copy)
            mask_frame = cv2.inRange(image_hsv, self._frame_hsv_lower,
                                     self._frame_hsv_upper)
            # 黑白反轉
            # frame_area = cv2.bitwise_not(frame_area, frame_area)
            if color_type == 'original':
                frame = cv2.bitwise_and(image_copy,
                                        image_copy,
                                        mask=mask_frame)
                return frame
            elif color_type == 'gray':
                image_gray = self._BGR_to_GRAY(image_copy)
                image_mode = self._get_mode(image_gray)
                image_gray[mask_frame == 0] = image_mode[-1]
                return image_gray
            else:
                return mask_frame
        else:
            raise ValueError('Please check your image!')

    def _pollution_area(self, image: np.ndarray,
                        color_type: str) -> np.ndarray:
        '''取出圖片pollution部分

        Args:
            image: cv2 np array的資料格式
            color_type: image的color型態

        Returns: 
            pollution_area: pollution部分的資料格式
        '''
        if isinstance(image, np.ndarray) and isinstance(color_type, str):
            image_copy = image.copy()
            image_hsv = self._BGR_to_HSV(image_copy)
            mask_pollution = cv2.inRange(image_hsv, self._pollution_hsv_lower,
                                         self._pollution_hsv_upper)
            # 黑白反轉
            # pollution_area = cv2.bitwise_not(pollution_area, pollution_area)
            if color_type == 'original':
                pollution = cv2.bitwise_and(image_copy,
                                            image_copy,
                                            mask=mask_pollution)
                return pollution
            elif color_type == 'gray':
                image_gray = self._BGR_to_GRAY(image_copy)
                # image_mode = self._get_mode(image_gray)
                # image_gray[pollution_mask == 0] = image_mode[-1]
                image_gray[mask_pollution == 0] = 255
                return image_gray
            else:
                return mask_pollution
        else:
            raise ValueError('Please check your image content!')

    def _set_dilate(self, image: np.ndarray) -> np.ndarray:
        '''設定飽和係數，將圖片內較薄的線條或缺口補滿

        Args:
            image: np格式的圖片資料

        Returns: 轉為dilate後的圖片資料
        '''
        if isinstance(image, np.ndarray):
            kernel = np.ones((3, 3), np.uint8)
            dilating_img = cv2.dilate(image, kernel, iterations=1)
            return dilating_img
        else:
            raise ValueError('Please input right image type!')

    def _set_gaussian(self, image: np.ndarray) -> np.ndarray:
        '''設定高斯模糊，將圖片朦朧化

        Args:
            image: np格式的圖片資料

        Returns: 轉為Gaussian後的圖片資料
        '''
        if isinstance(image, np.ndarray):
            blur_img = cv2.GaussianBlur(image, (3, 3), 0)
            return blur_img
        else:
            raise ValueError('Please input right image type!')

    def _get_mode(self, image: np.ndarray) -> any:
        '''取得圖片中的眾數，常用於灰階圖形處理
        bincount()是統計非負整數的個數，不能統計浮點數
        counts表示index代表出現的數，counts[index]代表出現數的次數
        今要求counts[index] 排序後最大跟第二大的counts的index(代表眾數跟出現第二多次的數)
        最後一個元素是counts最大值的index ，倒數第二是二大
        以防圖片出現大量黑色面積出現大量黑色區塊的話，取第二多數否則就return原本的眾數

        Args:
            image: np格式的圖片資料

        Returns: 
            index: 圖片中的最大值
        '''
        if isinstance(image, np.ndarray):
            counts = np.bincount(image.flatten())
            counts_sort = np.argsort(counts)
            index = counts_sort[-1]
            if index <= 100:
                index = counts_sort[-2]
                return index
            return counts_sort
        else:
            raise ValueError('Please input right image type!')

    def _input_path_check(self, input_path: str) -> None:
        '''確認資料夾是否存在若不在則創建新的
        Args:
            folder_path: 欲確認之資料夾主路徑
            sub_path: 欲確認之輸出資料夾路徑
            data_dirs ?: 欲轉換的資料夾路徑，如果有則會照原路徑產生若無則跳過
        Returns: output path
        '''
        if isinstance(input_path, str):
            folder_root = []
            folder_dirs = []
            folder_images = queue.Queue()
            if os.path.isfile(input_path):
                folder_images.put(input_path)
            elif os.path.isdir(input_path):
                for root, dirs, files in os.walk(input_path):
                    folder_root.append(root)
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        folder_dirs.append(dir_path)
                    for file in files:
                        if file.endswith('.jpg') or file.endswith(
                                '.png') or file.endswith('.jpeg'):
                            file_path = os.path.join(root, file)
                            folder_images.put(file_path)
            else:
                raise ValueError(
                    f"Can't idetify path '{input_path}', please check input path!"
                )
            return folder_root, folder_dirs, folder_images
        else:
            raise ValueError('Please check your folder path value!')

    def _ouput_folder_check(self, output_path: str, type_path: str, data_dirs: list) -> str:
        '''確認資料夾是否存在若不在則創建新的
        
        Args:
            folder_path: 欲確認之資料夾主路徑
            sub_path: 欲確認之輸出資料夾路徑
            data_dirs ?: 欲轉換的資料夾路徑，如果有則會照原路徑產生若無則跳過
            
        Returns: output path
        '''
        # if os.path.exists(output_path):
        #     shutil.rmtree(output_path)
        #     self._logger.info(
        #         f"Previous prediction folder: {output_path} has been remove!!")
        if isinstance(output_path, str) and isinstance(type_path, str):
            output_folder_path = os.path.join(output_path, type_path)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                self._logger.info(
                    f"output folder: {output_folder_path} has been created!!")
            if len(data_dirs) > 0:
                for dir in data_dirs:
                    dir = dir.replace(f"{self._folder_root[0]}\\", "")
                    data_folder_dir = os.path.join(output_folder_path, dir)
                    if not os.path.exists(data_folder_dir):
                        os.makedirs(data_folder_dir)
                        self._logger.info(
                            f"output folder: {data_folder_dir} has been created!!"
                        )
            return output_folder_path
        else:
            raise ValueError('Please check your folder path value!')

    def _nothing(self, x):
        pass

    def get_image_transfer(
        self,
        image_path: str,
        output_type: str,
        output_more:bool = False
    ) -> dict:
        '''將圖片依照 彩色 灰階 黑白進行rotate, flip, rotate filp 處理後輸出二元圖 
        
        Args:
            image_path: 圖片的位置
            output_type: 輸出的圖片格式，預設格式='binary'可以選擇'gray'
            
        Returns: dict (處理後的圖片會存成dict回傳)
        '''
        try:
            transfer_dict = {}

            # read image
            image_original = self._cv_imread(image_path)

            # make image resize
            image_resize = self._image_resize(image_original)

            # get part of image pollution and frame
            pollution = self._pollution_area(image_resize, output_type)
            frame = self._frame_area(image_resize, output_type)
            # cv2.imshow('pollution', pollution)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

            # output image dict name: 1->original 2->rotate, 3->flip, 4->flip and rotate
            transfer_dict["1"] = self._image_merge(pollution, frame)
            # make image rotate, flip
            if output_more == True:
                rotate_image = self._image_rotate(pollution)
                flip_image = self._image_flip(pollution)
                rotate_flip_image = self._image_rotate(flip_image)
                transfer_dict["2"] = self._image_merge(rotate_image, frame)
                transfer_dict["3"] = self._image_merge(flip_image, frame)
                transfer_dict["4"] = self._image_merge(rotate_flip_image, frame)

            if output_type != 'original':
                for num, image in transfer_dict.items():
                    # 黑白反轉
                    transfer_dict[num] = cv2.bitwise_not(image, image)

            return transfer_dict

        except Exception as e:
            self._logger.error(e)

        # 另一個寫法 use imwrite mode -> can't support chinese file name
        # cv2.imwrite("L201030128_Total_35_1.jpg", merge_original_image,
        #             [cv2.IMWRITE_JPEG_QUALITY, 90])
        # cv2.imwrite('L201030128_Total_35_1.png', merge_original_image,
        #             [cv2.IMWRITE_PNG_COMPRESSION, 5])

    def get_image_output(self, thread_num: str, output_type: str,
                         output_path: str, output_more:bool):
        '''將處理後的圖片輸出 
        Args:
            image_path: 圖片的位置
            output_type: 輸出的圖片格式，預設格式='binary'可以選擇'gray'
            output_path: 輸出的圖片位置(預設路徑=/output，若要自行設定請以絕對路徑輸入)
        Returns: dict
        '''
        try:
            # self._ouput_folder_check(output_path, output_type)
            while self._image_queue.qsize() > 0:
                image_root = self._folder_root
                image_path = self._image_queue.get()
                image_name = image_path.split('.')[0]
                image_type = image_path.split('.')[1]
                if len(image_root) > 0:
                    image_name = image_name.replace(f"{image_root[0]}\\","")
                print(
                    f"Thread{thread_num} has started to process image '{image_path}' to folder '{output_path}'"
                )
                transfer_dict = self.get_image_transfer(
                    image_path, output_type, output_more)
                for num, image in transfer_dict.items():
                    if image_type != "png":
                        cv2.imencode('.jpg', image)[1].tofile(
                            os.path.join(output_path,
                                         f"{image_name}_{num}.jpg"))
                    else:
                        cv2.imencode('.png', image)[1].tofile(
                            os.path.join(output_path,
                                         f"{image_name}_{num}.png"))
        except Exception as e:
            self._logger.error(e)

    def get_hsv_value(self, image_path: str) -> dict:
        '''調整圖片的HSV值，供參考

        Args:
            image_path: 圖片的位置

        Returns:
            hsv_check_result: 調整後的HSV值
        '''
        try:
            windows_name = f"GET {image_path} HSV"
            img = self._cv_imread(image_path)
            self.BGR = img
            self.GRAY = self._BGR_to_GRAY(img)
            self.HSV = self._BGR_to_HSV(img)
            cv2.namedWindow(windows_name, self._windows_size)
            cv2.setMouseCallback(windows_name, self._mouse_click)
            while True:
                cv2.imshow(windows_name, img)
                if cv2.waitKey(0) & 0xFF == ord(
                        'q') or cv2.waitKey(0) & 0xFF == 27:  # Esc
                    break
            result = self.mouse_click
            cv2.destroyAllWindows()
            return result
        except Exception as e:
            self._logger.error(e)

    def adjust_hsv_value(self, image_path: str, area: str) -> dict:
        '''調整圖片的HSV值，供參考

        Args:
            image_path: 圖片的位置
            area: 調整的區域，可選取"frame"或"pullotion"

        Returns:
            hsv_check_result: 調整後的HSV值
        '''
        try:
            windows_name = f"ADJUST {image_path} HSV VALUE"
            # read image
            image_original = self._cv_imread(image_path)
            # make image resize
            image_resize = self._image_resize(image_original)
            # make image from RGB to HSV
            image_hsv = self._BGR_to_HSV(image_resize)

            cv2.namedWindow(windows_name, self._windows_size)
            color_range = self._get_color_range(area)
            for color_name, color_value in color_range.items():
                cv2.createTrackbar(color_name, windows_name, color_value, 255,
                                   self._nothing)
            hsv_check_result = {}
            while (True):
                lowerbH = cv2.getTrackbarPos('LowerbH', windows_name)
                lowerbS = cv2.getTrackbarPos('LowerbS', windows_name)
                lowerbV = cv2.getTrackbarPos('LowerbV', windows_name)
                upperbH = cv2.getTrackbarPos('UpperbH', windows_name)
                upperbS = cv2.getTrackbarPos('UpperbS', windows_name)
                upperbV = cv2.getTrackbarPos('UpperbV', windows_name)

                image_target = cv2.inRange(image_hsv,
                                           (lowerbH, lowerbS, lowerbV),
                                           (upperbH, upperbS, upperbV))
                image_specifiedColor = cv2.bitwise_and(image_resize,
                                                       image_resize,
                                                       mask=image_target)
                cv2.imshow(windows_name, image_specifiedColor)
                if cv2.waitKey(1) & 0xFF == ord(
                        'q') or cv2.waitKey(1) & 0xFF == 27:  # Esc:
                    hsv_check_result["lowerbH"] = lowerbH
                    hsv_check_result["lowerbS"] = lowerbS
                    hsv_check_result["lowerbV"] = lowerbV
                    hsv_check_result["lpperbH"] = upperbH
                    hsv_check_result["lpperbS"] = upperbS
                    hsv_check_result["lpperbV"] = upperbV
                    break

            cv2.destroyAllWindows()
            return hsv_check_result
        except Exception as e:
            self._logger.error(e)

    def run(self,
            input_path: str,
            output_type: str,
            output_path: str = 'output',
            output_more:bool = False):
        '''
        執行跑image transfer and image ouput，可mutithreading
        Args:
            input_path: 圖片的位置，可以是圖片或是目錄
            output_type: 輸出圖片的類型，可選取"original"、"gray"或是"binary"
            output_path: 輸出圖片的位置，預設路徑為"output"
        Returns: None
        '''
        try:
            thread_list = []
            self._folder_root, self._folder_list, self._image_queue = self._input_path_check(
                input_path)
            output_folder_path = self._ouput_folder_check(
                output_path, output_type, self._folder_list)
            for i in range(self._thread_num):
                process = threading.Thread(target=self.get_image_output,
                                           args=(str(i), 
                                                 output_type,
                                                 output_folder_path,
                                                 output_more),
                                           daemon=True)
                thread_list.append(process)

            for i in range(self._thread_num):
                thread_list[i].start()

            for i in range(self._thread_num):
                thread_list[i].join()

        except Exception as e:
            self._logger.error(e)

    # def image_test_(self, image_path: str) -> None:
    #     '''
    #     測試用
    #     '''
    #     # read image
    #     image_original = self._cv_imread(image_path)

    #     # make image resize
    #     image_resize = self._image_resize(image_original)

    #     cv2.imshow('image_resize', image_resize)
    #     image_hsv = self._BGR_to_HSV(image_resize)

    #     # 从HSV图像中提取色调通道
    #     mask_pollution = cv2.inRange(image_hsv, self._pollution_hsv_lower,
    #                                  self._pollution_hsv_upper)
    #     mask_frame = cv2.inRange(image_hsv, self._frame_hsv_lower,
    #                              self._frame_hsv_upper)

    #     # image_mask = image_resize.copy()
    #     # image_original_rotate = cv2.rotate(image_resize, cv2.ROTATE_180)
    #     # mask_pollution_rotate = cv2.rotate(mask_pollution, cv2.ROTATE_180)

    #     # image_original_flip = cv2.flip(image_resize, 1)
    #     # mask_pollution_flip = cv2.rotate(mask_pollution, cv2.ROTATE_180)

    #     pollution = cv2.bitwise_and(image_resize,
    #                                 image_resize,
    #                                 mask=mask_pollution)
    #     pollution_rotate = cv2.rotate(pollution, cv2.ROTATE_180)
    #     frame = cv2.bitwise_and(image_resize, image_resize, mask=mask_frame)
    #     cv2.imshow('pollution', pollution_rotate)
    #     cv2.imshow('frame', frame)
    #     image_merge = cv2.bitwise_xor(pollution_rotate, frame)
    #     cv2.imshow('image_merge', image_merge)
    #     cv2.waitKey(0)


time_1 = time.time()
imagePath = 'L201030128_金面汙染_Total_35.png'
imagePath1 = 'C:\\Users\\k09857\\Desktop\\idenprof\\au_pollution_defect_images\\20220321_exec_images\\first_expriment\\train\\panel'
imagePath2 = 'C:\\Users\\k09857\\Desktop\\idenprof\\au_pollution_detect_resource\\au_defect_experiment_20220401\\train\\strip'
output = ['original', 'gray', 'binary']
b = ImageProcess()
# result = b.adjust_hsv_value(filePath, "frame")
# result = b.get_hsv_value(filePath)
# b.get_image_transfer(imagePath, output[2])
# b.image_test(filePath)
# b.get_image_output(imagePath, output[2])
for i in range(len(output)):
    b.run(imagePath, output[i])

time_2 = time.time()
time_interval = time_2 - time_1
print("time: ", time_interval)