import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np

class BookKeeper():
    def __init__(self, batches_per_epoch, batch_report_gap=200): 
        
        # params
        self.batches_per_epoch = batches_per_epoch 
        self.batch_report_gap = batch_report_gap

        # graphs
        self.lists = {
            'batch_loss': {'batch': [], 'value': []},
            'train_loss': {'batch': [], 'value': []},
            'test_loss': {'batch': [], 'value': []},
            'test_r2': {'batch': [], 'value': []}, 
            'actual_r2': {'batch': [], 'value': []}, 
        }

        # timer
        self.batches_completed = 0
        self.epochs_completed = 0


    ######################################## 
    # GRAPHS
    ########################################

    def append(self, key, val):
        self.lists[key]['batch'].append(self.batches_completed)
        self.lists[key]['value'].append(val)

    def show_loss(self):
        plt.figure(figsize=(10,3))
        plt.title('losses')
        plt.xticks(np.arange(0, np.array(self.lists['train_loss']['batch']).max()+self.batches_per_epoch, 
                             self.batches_per_epoch))
        plt.plot(self.lists['train_loss']['batch'][1:], 
                 self.lists['train_loss']['value'][1:],
                 label='train loss')
        plt.plot(self.lists['test_loss']['batch'][1:], 
                 self.lists['test_loss']['value'][1:],
                 label='test loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_plots(self):
        self.show_loss()

        for k in ['batch_loss', 'test_r2', 'actual_r2']:
            data = self.lists[k]['value']
            plt.figure(figsize=(10,3))
            plt.title(k)
            plt.xticks(np.arange(0, np.array(self.lists[k]['batch']).max()+self.batches_per_epoch, 
                                 self.batches_per_epoch))
            # plt.ylim(0, np.mean(data) + 3 * np.std(data))
            plt.plot(self.lists[k]['batch'][1:], data[1:]) #ignore 1st entry, too big
            plt.grid(True)
            plt.show()

    ######################################## 
    # CLOCK 
    ########################################


    def tick_batch(self, batch_loss):
        self.batches_completed += 1
        self.append('batch_loss', batch_loss)

        if self.batches_completed % self.batch_report_gap == 0:
            self.report_batch(batch_loss = np.mean(self.lists['batch_loss']['value'][:-10]))

    def report_batch(self, batch_loss):
        epochs_completed = self.batches_completed // self.batches_per_epoch
        batches_completed = self.batches_completed % self.batches_per_epoch
        print('Epoch', f"{epochs_completed : .2f}",
              'Batch', f"{batches_completed : 04}",
              batch_loss, 
              datetime.fromtimestamp(time.time()).strftime("%H:%M:%S"))

    def tick_epoch(self, train_loss, test_loss, transformed_r2, actual_r2):
        # epochs_completed = self.batches_completed // self.batches_per_epoch
        self.epochs_completed += 1

        self.append('train_loss', train_loss)
        self.append('test_loss', test_loss)
        self.append('test_r2', transformed_r2)
        self.append('actual_r2', actual_r2)
        
        print('### Epoch', f"{self.epochs_completed : .2f}", '|',
              'train_loss', train_loss, 
              'test_loss', test_loss, 
              'test_r2', transformed_r2, 
              'actual_r2', actual_r2)
        print()
