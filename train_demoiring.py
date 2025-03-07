import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser, Namespace
from pathlib import Path
from config import Config
from image_preprocess import Image_preprocessing, Image_preprocessing_test
from dataset import Image_dataset
from cnn_model import Image_model
from unet import UNet
from sklearn.preprocessing import MinMaxScaler
import pickle
import torchmetrics
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import random
from util import AverageMeter
import gc

from trainer import Trainer


def main(args):

    config = Config()
    seed_everything(config.SEED)


    g_model_ckpt_dir = args.ckpt_dir / f"model_generator_phase1"
    g_model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    d_model_ckpt_dir = args.ckpt_dir / f"model_discriminator_phase1"
    d_model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    g_model_ckpt_dir_phase2 = args.ckpt_dir / f"model_generator_phase2"
    g_model_ckpt_dir_phase2.mkdir(parents=True, exist_ok=True)

    d_model_ckpt_dir_phase2 = args.ckpt_dir / f"model_discriminator_phase2"
    d_model_ckpt_dir_phase2.mkdir(parents=True, exist_ok=True)

    g_model_ckpt_dir_phase3 = args.ckpt_dir / f"model_generator_phase3_with_freq_loss"
    g_model_ckpt_dir_phase3.mkdir(parents=True, exist_ok=True)

    d_model_ckpt_dir_phase3 = args.ckpt_dir / f"model_discriminator_phase3_with_freq_loss"
    d_model_ckpt_dir_phase3.mkdir(parents=True, exist_ok=True)

    d_model_pretrain_ckpt_dir = args.ckpt_dir / f"model_pretrain_discriminator"

    gc.collect()
    torch.cuda.empty_cache()


    # image path
    moire_image_path = config.moire_data_path
    clean_image_path = config.clean_data_path
    test_moire_image_path = config.test_moire_data_path
    test_clean_image_path = config.test_clean_data_path


  # image data preprocessing
    img_preprocess = Image_preprocessing(moire_image_path, clean_image_path)
    (total_image_arr,  total_label_arr) = img_preprocess.preprocessing()

    img_preprocess_test = Image_preprocessing_test(test_moire_image_path, test_clean_image_path)
    (test_moire_image_arr,  test_clean_image_arr, test_moire_label, test_clean_label) = img_preprocess_test.preprocessing()


    # split trainset, valset, testset
    X_train, X_val, y_train, y_val = train_test_split(total_image_arr, total_label_arr, test_size=0.20, random_state = config.SEED)
    #X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state = config.SEED)
    

    # Normalize image data
    img_size = config.img_size

    X_train_std = X_train / 255
    X_val_std = X_val / 255
    X_test_std = test_moire_image_arr / 255



    # Store testing data
    with open(os.path.join('data_ckpt', 'test_Moire.pickle'), 'wb') as handle:
        pickle.dump(X_test_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_moire_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join('data_ckpt', 'test_Clean.pickle'), 'wb') as handle:
        pickle.dump(test_clean_image_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_clean_label, handle, protocol=pickle.HIGHEST_PROTOCOL)



    # Make image dataset
    train_image_dataset = Image_dataset(X_train_std, y_train)
    val_image_dataset = Image_dataset(X_val_std, y_val)

    # put into dataloader
    train_image_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size = config.TRAIN_BS, shuffle = False)
    val_image_dataloader = torch.utils.data.DataLoader(val_image_dataset, batch_size = config.VALID_BS, shuffle = False)






    if args.first_phase: 
        print("----------------------------------------------First Phase----------------------------------------")
        # define model
        g_model = UNet(n_channels = 1, n_classes = 1)
        d_model = Image_model()

        if config.discriminator_use_pretrain:
            # load pretrain discriminator
            d_pretrain_ckpt_model = torch.load(os.path.join(d_model_pretrain_ckpt_dir, "best-model.pth"))
            d_model.load_state_dict(d_pretrain_ckpt_model)

        if config.use_wandb:
            print('use wandb')
            import wandb
            wandb.init(project='Thesis_demoire_phase1', config=args)

        first_phase_trainer = Trainer(g_first_phase_model = g_model, 
                                      d_first_phase_model = d_model, 
                                      train_image_dataloader = train_image_dataloader, 
                                      val_image_dataloader = val_image_dataloader, 
                                      epochs = config.EPOCHS,
                                      phase = 1,
                                      g_model_ckpt_dir = g_model_ckpt_dir, 
                                      d_model_ckpt_dir = d_model_ckpt_dir)
        
        first_phase_trainer.start_training()

        
    ###
    # Second phase of training

    if args.second_phase:
        print("----------------------------------------------Second Phase----------------------------------------")
        if config.use_wandb:
            print('use wandb')
            import wandb
            wandb.init(project='Thesis_demoire_phase2', config=args)



        # define model
        g_first_phase_model = UNet(n_channels = 1, n_classes = 1)
        d_first_phase_model = Image_model()


        g_first_phase_ckpt_model = torch.load(os.path.join(g_model_ckpt_dir, "final-model.pth"))
        d_first_phase_ckpt_model = torch.load(os.path.join(d_model_ckpt_dir, "final-model.pth"))


        g_first_phase_model.load_state_dict(g_first_phase_ckpt_model)
        d_first_phase_model.load_state_dict(d_first_phase_ckpt_model)
        
        second_phase_trainer = Trainer(
                                g_first_phase_model = g_first_phase_model, 
                                d_first_phase_model = d_first_phase_model, 
                                train_image_dataloader = train_image_dataloader, 
                                val_image_dataloader = val_image_dataloader, 
                                epochs = config.EPOCHS_2,
                                phase = 2,
                                g_model_ckpt_dir = g_model_ckpt_dir_phase2, 
                                d_model_ckpt_dir = d_model_ckpt_dir_phase2)
        
        second_phase_trainer.start_training()

    ###
    # Third phase of training
    # load pretrain discriminator
    if args.third_phase:
        print("----------------------------------------------Third Phase----------------------------------------")
        if config.use_wandb:
            print('use wandb')
            import wandb
            wandb.init(project='Thesis_demoire_phase3', config=args)

        # define model
        g_second_phase_model = UNet(n_channels = 1, n_classes = 1)
        d_second_phase_model = Image_model()

        g_second_phase_ckpt_model = torch.load(os.path.join(g_model_ckpt_dir_phase2, "final-model.pth"))
        d_second_phase_ckpt_model = torch.load(os.path.join(d_model_ckpt_dir_phase2, "final-model.pth"))

        g_second_phase_model.load_state_dict(g_second_phase_ckpt_model)
        d_second_phase_model.load_state_dict(d_second_phase_ckpt_model)

        third_phase_trainer = Trainer(
                                g_first_phase_model = g_second_phase_model, 
                                d_first_phase_model = d_second_phase_model, 
                                train_image_dataloader = train_image_dataloader, 
                                val_image_dataloader = val_image_dataloader, 
                                epochs = config.EPOCHS_3,
                                phase = 3,
                                g_model_ckpt_dir = g_model_ckpt_dir_phase3, 
                                d_model_ckpt_dir = d_model_ckpt_dir_phase3)
        
        third_phase_trainer.start_training()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =True



def parse_args() -> Namespace:
    parser = ArgumentParser()

    # path
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./model_ckpt",
    )

    parser.add_argument(
        "--first_phase",  
        action = "store_true",
        default=False,
    )

    parser.add_argument(
        "--second_phase",
        action = "store_true",
        default=False,
    )

    parser.add_argument(
        "--third_phase",
        action = "store_true",
        default=False,
    )


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
