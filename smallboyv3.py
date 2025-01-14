from bookkeeping import *
from data_processing import *

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import os


vgg_output_size = 2048
fc_output_size = 64
num_predictions = 6

# learn_rate = 0.0005
# learn_decay = 0.9


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()

        self.cnn = nn.Sequential(
            # - Conv(003, 064, 3, 1, 1) - BatchNorm(064) - ReLU - MaxPool(2, 2)
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # - Conv(064, 128, 3, 1, 1) - BatchNorm(128) - ReLU - MaxPool(2, 2)
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # - Conv(128, 256, 3, 1, 1) - BatchNorm(256) - ReLU
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # # - Conv(256, 256, 3, 1, 1) - BatchNorm(256) - ReLU - MaxPool(2, 2)
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # - Conv(256, 512, 3, 1, 1) - BatchNorm(512) - ReLU
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            # - Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # - Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            # - Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(512*4*4, vgg_output_size)
        )

    def forward(self, input):

        return self.cnn(input)
    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(163, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),

            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),

            nn.Linear(128, fc_output_size), nn.ReLU(inplace=True), nn.Dropout(0.2),

            # nn.Linear(128, fc_output_size), nn.ReLU(inplace=True), nn.Dropout(0.2),

            # nn.Linear(256, fc_output_size)
        )

    def forward(self, input):
        return self.linear(input)

class SmallBoyV3(nn.Module):
    def __init__(self):
        super(SmallBoyV3, self).__init__()

        self.vgg = VGG11()
        self.fc = FC()

        concat_size = vgg_output_size + fc_output_size
        self.linear_head = nn.Sequential(
            nn.Linear(concat_size, concat_size//2), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(concat_size//2, concat_size), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(concat_size, num_predictions),
            # nn.Sigmoid()
        )

    def forward(self, img, data):
        vgg_result = self.vgg(img)
        data_result = self.fc(data)

        combined = torch.cat((vgg_result, data_result), dim=1)

        return self.linear_head(combined)
    

# class Model(nn.Module):

#     def __init__(self, device, batches_per_epoch, dataprocessor):
#         super(Model, self).__init__()
#         self.device = device 
#         self.model = SmallBoyV3()

#         self.dp = dataprocessor 

#         self.optimizer = optim.AdamW(self.model.parameters(), lr=learn_rate, weight_decay=0.02)
#         self.lambda_lr = lambda step: learn_decay ** step
#         self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_lr)
#         self.criterion = nn.MSELoss()

#         self.bk = BookKeeper(batches_per_epoch)

#     def forward(self, img, data):
#         return self.model(img, data)


#     def train_one_epoch(self, train_dataloader, test_dataloader):
#         self.model.train()

#         running_loss = []

#         for batch_idx, (images, data, target) in enumerate(train_dataloader):
            
#             self.optimizer.zero_grad()

#             images = images.to(self.device)
#             data = data.to(self.device)
#             target = target.to(self.device)

#             prediction = self.forward(images, data)
#             loss = self.criterion(prediction, target)

#             loss.backward()
#             self.optimizer.step()

#             self.bk.tick_batch(loss.item())
#             running_loss.append(loss.item())

#             if batch_idx > len(train_dataloader)//3:
#                 break # just train on a third of shuffled data every epoch

#         self.scheduler.step() # STEPPING SCHEDULER 3x PER EPOCH

#         test_loss, transformed_r2, actual_r2 = self.validate(test_dataloader)
#         train_loss = np.array(running_loss).mean()
#         self.bk.tick_epoch(train_loss, test_loss, transformed_r2, actual_r2)

#         self.bk.show_plots() # plot every epoch

#         # if np.abs(r2) < 100:
#         #     torch.save(self.model.state_dict(), os.path.join('data', f"small_boy_epoch{epoch}.sav"))
#         #     print('saved', f"small_boy_epoch{epoch}.sav", 'r2 = ', r2)
#         #     print('lr' , get_current_lr(optimizer))

#         #     predict(model)

#         return train_loss, test_loss, transformed_r2, actual_r2

#     def validate(self, test_dataloader):
#         self.model.eval()

#         test_loss = []
#         acc_predictions = [] # hold onto predictions and targets for R2
#         acc_targets = []

#         with torch.no_grad():
#             for images, data, target in test_dataloader:

#                 images = images.to(self.device)
#                 data = data.to(self.device)
#                 target = target.to(self.device)

#                 prediction = self.forward(images, data)
#                 loss = self.criterion(prediction, target).item()
#                 test_loss.append(loss)

#                 acc_predictions.append(prediction)
#                 acc_targets.append(target)

#         test_loss = np.mean(test_loss) # / len(test_dataloader)

#         acc_targets_cpu = [tensor.cpu().numpy() for tensor in acc_targets]
#         acc_predictions_cpu = [tensor.cpu().numpy() for tensor in acc_predictions]
#         predictions = np.concatenate(acc_predictions_cpu)
#         transformed_r2 = r2_score(np.concatenate(acc_targets_cpu), predictions)

#         target_df = pd.DataFrame(predictions, columns=TRAIN_COLUMN_ORDER)
#         pred_df = pd.DataFrame(predictions, columns=TRAIN_COLUMN_ORDER)
#         actual_pred_df = self.dp.inv_transform_Y(pred_df)
#         actual_r2 = r2_score(target_df, actual_pred_df)

#         return test_loss, transformed_r2, actual_r2

#     def predict(self, dataloader):
#         self.model.eval()
#         predictions = []

#         with torch.no_grad():
#             for images, data, _ in dataloader:
#                 prediction = self.forward(images, data).detach().cpu().numpy()
#                 predictions.append(prediction) # X4,X11,X18,X26,X50,X3112

#         all_predictions_np = np.concatenate(predictions, axis=0)
#         df = pd.DataFrame(all_predictions_np, columns=TRAIN_COLUMN_ORDER)
#         # df = df[['X4', 'X11', 'X18', 'X50', 'X26', 'X3112']] # format to id,X4,X11,X18,X50,X26,X3112

#         # final_df = pd.concat([img_id, df], axis=1)

#         return df
    
#     def predict_with_imgid(self, img_id, dataloader):
#         df = self.predict(dataloader)
#         return pd.concat([img_id, df], axis=1)
    