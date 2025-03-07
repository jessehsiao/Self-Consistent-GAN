import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.nn.modules.loss import BCEWithLogitsLoss

class Config:
    def __init__(self):
        self.EPOCHS = 150
        self.EPOCHS_2 = 160
        self.EPOCHS_3 = 360
        self.DEVICE = torch.device('cuda')

        self.SEED = 127
        
        self.phase2_w1 = 1/2
        self.phase2_w2 = 1/2

        self.phase3_w1 = 0.2
        self.phase3_w2 = 0.2
        self.phase3_w3 = 0.45
        self.phase3_w4 = 0.15
        self.freq_loss = True

        self.moire_data_path = os.path.join("..", "data", 'data_simulation', 'Final_image_train', 'Final_image_train_Moire')
        self.clean_data_path = os.path.join("..", "data", 'data_simulation', 'Final_image_train', 'Final_image_train_Clean')
        self.test_moire_data_path = os.path.join("..", "data", 'data_simulation', 'Final_image_test', 'Final_image_test_Moire')
        self.test_clean_data_path = os.path.join("..", "data", 'data_simulation', 'Final_image_test', 'Final_image_test_Clean')
        

        self.img_size = 256
        self.TRAIN_BS = 32
        self.VALID_BS = 32
        self.TEST_BS = 32
        self.type_scheduler = 'cycle'
        #self.use_pretrained = True
        #self.DATA_DATE = "20220826"
        self.model_type = 'cnn1'

        self.first_phase = True
        self.second_phase = False
        self.third_phase = False

        self.fix_imbalanced = False
        #self.g_loss_fn = torch.nn.MSELoss()
        self.loss_fn = BCEWithLogitsLoss()
        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn_constrain = torch.nn.MSELoss(reduction='mean')

        #self.data_path = os.path.join("..", "data", '20221003', '67MOT01692_Epoxy_0930R')
        self.g_lr = 3e-4
        self.d_lr = 3e-4
        #self.corner = 'TR'
        #self.pred_target = 'Side(um)'

        self.g_type = "unet"
        self.test_model = "final-model.pth"
        self.image_result_date = "20230717_model_generator_Unet_phase3_with_freq"
        self.discriminator_use_pretrain = False

        self.use_wandb = True

        self.clamp_num = 0.01 # For WGAN 

        #### phase 2 best model 效果較好