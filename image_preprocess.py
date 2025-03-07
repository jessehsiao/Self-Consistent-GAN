import numpy as np
import os 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from config import Config

class Image_preprocessing():
    def __init__(self, moire_path, clean_path):
        self.moire_path = moire_path
        self.clean_path = clean_path

    def preprocessing(self):

        config = Config()
        # moire image
        # 1. read all image 2. turn to grey channel 3.  add to np array
        transform = T.Resize((config.img_size, config.img_size))
        moire_img_arr = np.array([np.array(transform(Image.open(os.path.join(self.moire_path, i))).convert('L')) for i in tqdm(os.listdir(self.moire_path))])
        moire_label = np.array([1 for i in range(moire_img_arr.shape[0])])
        # clean image
        clean_img_arr = np.array([np.array(transform(Image.open(os.path.join(self.clean_path, i))).convert('L')) for i in tqdm(os.listdir(self.clean_path))])
        clean_label = np.array([0 for i in range(clean_img_arr.shape[0])])

        total_image_arr = np.append(moire_img_arr, clean_img_arr, axis=0)
        total_label_arr = np.append(moire_label, clean_label, axis=0)


        return (total_image_arr, total_label_arr)
    


class Image_preprocessing_test():
    def __init__(self, moire_path, clean_path):
        self.moire_path = moire_path
        self.clean_path = clean_path


    def preprocessing(self):

        config = Config()
        transform = T.Resize((config.img_size, config.img_size))

        count = 0
        for image_file in tqdm(os.listdir(self.clean_path)):
            count += 1


        moire_img_arr = np.array([np.array(transform(Image.open(os.path.join(self.moire_path, f"test_Moire_image_no{i}.png"))).convert('L')) for i in range(count)])
        moire_label = np.array([1 for i in range(moire_img_arr.shape[0])])

        # clean image
        clean_img_arr = np.array([np.array(transform(Image.open(os.path.join(self.clean_path, f"test_Clean_image_no{i}.png"))).convert('L')) for i in range(count)])
        clean_label = np.array([0 for i in range(clean_img_arr.shape[0])])

        #total_image_arr = np.append(moire_img_arr, clean_img_arr, axis=0)
        #total_label_arr = np.append(moire_label, clean_label, axis=0)


        return (moire_img_arr, clean_img_arr, moire_label, clean_label)