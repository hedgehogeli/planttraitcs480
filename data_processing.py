
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
# 
import torchvision
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


MEAN = [113.82422637939453/255, 114.86695861816406/255, 85.6895751953125/255]
STD = [46.77458190917969/255, 45.75661849975586/255, 45.359466552734375/255]

augment_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

standard_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


class DataProcessor():
    def __init__(self):
        super().__init__()

        self.train_data = pd.read_csv(os.path.join('data', 'train.csv'))
        self.eval_data = pd.read_csv(os.path.join('data', 'test.csv'))

        self.x_outlier_cutoff = 5
        self.y_outlier_cutoff = 3

    def get_and_process_train_data(self):

        X = self.train_data[self.train_data.columns[1:164]]
        Y = self.train_data[self.train_data.columns[164:]]

        self.get_transform_params_X(X)
        X_NORMALISED = self.normalise_data(X)
        X_CLEANED, x_dropped_indices = self.clean_X(X_NORMALISED)

        self.get_transform_params_Y(Y)
        Y_LOG_NORMALISED = self.transform_Y(Y)
        Y_CLEANED, y_dropped_indices = self.clean_Y(Y_LOG_NORMALISED) 

        X_CLEANED.drop([i for i in y_dropped_indices if i not in x_dropped_indices], inplace=True) 
        X_CLEANED.reset_index(drop=True, inplace=True)
        Y_CLEANED.drop([i for i in x_dropped_indices if i not in y_dropped_indices], inplace=True) 
        Y_CLEANED.reset_index(drop=True, inplace=True)
        # 2299 rows dropped total

        X_IMG_IDS = self.train_data[self.train_data.columns[0:1]].drop(set(x_dropped_indices+y_dropped_indices))
        X_IMG_IDS.reset_index(drop=True, inplace=True)

        return X_IMG_IDS, X_CLEANED, Y_CLEANED

    def get_and_process_eval_data(self):
        X_EVAL = self.eval_data[self.eval_data.columns[1:]]
        X_EVAL_NORMALISED = self.normalise_data(X_EVAL)

        X_EVAL_IMG_IDS = self.eval_data[self.eval_data.columns[0:1]]

        return X_EVAL_IMG_IDS, X_EVAL_NORMALISED
    
    ########################################
    # X (FEATURE) DATA PROCESSING:
    ########################################

    def get_transform_params_X(self, X):
        self.X_means = X.mean()
        self.X_stds = X.std()

    def normalise_data(self, X):
        """ Convert 'data' into Z-score """
        return (X - self.X_means) / self.X_stds

    def clean_X(self, X):
        """ 
        Drop X values greater than self.x_outlier_cutoff=5 stdevs away from mean 
        Input is already in Z score.
        Return new df, list of indices dropped
        """
        rows_to_drop = X[(np.abs(X) > self.x_outlier_cutoff).any(axis=1)]
        indices_to_drop = rows_to_drop.index
        new_Y = X.drop(indices_to_drop)

        return new_Y, indices_to_drop.tolist()

    ########################################
    # Y (LABEL) DATA PROCESSING:
    ########################################

    def get_transform_params_Y(self, Y):
        self.Y_mins = np.min(Y, axis=0)
        self.Y_means = np.mean(np.log10(Y - self.Y_mins + 1e-6), axis=0)
        self.Y_stds = np.std(np.log10(Y - self.Y_mins + 1e-6), axis=0)

    def transform_Y(self, Y):
        """ Transform LABELS by their log and then Z score. """
        Y_zerod = Y - self.Y_mins
        log_Y = np.log10(Y_zerod + 1e-6)
        normalized_Y = (log_Y - self.Y_means) / self.Y_stds
        # standardized_Y = (normalized_Y + self.y_outlier_cutoff) / (2 * self.y_outlier_cutoff)
        standardized_Y = normalized_Y
        return standardized_Y

    def inv_transform_Y(self, Y):
        """ Restore labels """
        # normalized_Y = Y * (2 * self.y_outlier_cutoff) - self.y_outlier_cutoff
        normalized_Y = Y
        original_Y = 10 ** (normalized_Y * self.Y_stds + self.Y_means) + self.Y_mins
        return original_Y
    
    def clean_Y(self, Y):
        """ 
        Drop Y values that are more than 
        self.y_outlier_cutoff stdevs away from mean
        Input is already in Z score.
        Return new df, list of indices dropped
        """
        # indices_to_drop = Y.index[(Y > 1).any(axis=1)].append(Y.index[(Y < 0).any(axis=1)])
        indices_to_drop = Y.index[(np.abs(Y) > self.y_outlier_cutoff).any(axis=1)]
        new_Y = Y.drop(indices_to_drop)
        return new_Y, indices_to_drop.tolist()

    
    ########################################
    # DATA VISUALISATIONS
    ########################################

    def hist(self, data, bins=50):
        plt.figure(figsize=(10,2))
        plt.hist(data, bins=bins)
        plt.show()


    
class CustomDataset(Dataset):
    def __init__(self, img_id, data, target, img_dir, transform=None):
        self.img_id = img_id
        self.data_frame = data
        self.target = target
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.img_id['id'][idx]}.jpeg")
        image = np.array(plt.imread(img_name), dtype=np.float32)
        if self.transform:
            image = self.transform(image)

        training_data = self.data_frame[idx]
        noise = np.random.normal(0, 0.005, training_data.shape)
        training_data = torch.tensor(training_data + noise)
        target_data = torch.tensor(self.target[idx])

        return image, training_data, target_data
    





# all_img_id = train_data['id']
# full_dataset = CustomDataset(all_img_id, X, Y, img_dir=TRAIN_IMG_DIR, transform = standard_transform)
# full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False)

# mean = 0.0
# std = 0.0
# total_images = 0

# for images, _, _ in full_dataloader:
#     images = images.view(images.size(0), images.size(1), -1)  # Reshape to (batch_size, channels, height*width)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)
#     total_images += images.size(0)

# mean /= total_images
# std /= total_images

# mean.tolist()
# std.tolist()