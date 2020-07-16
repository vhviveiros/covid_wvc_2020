# This work is based on https://github.com/kapil-varshney/utilities/blob/master/training_plot/trainingplot.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.keras
from utils import check_folder
# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
matplotlib.use('agg')


class TrainingPlot(tensorflow.keras.callbacks.Callback):

    def __init__(self, total_epochs):
        self.epochs = total_epochs

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.loss = []
        self.acc = []
        self.precision = []
        self.auc = []
        self.rec = []

        self.val_loss = []
        self.val_acc = []
        self.val_precision = []
        self.val_auc = []
        self.val_rec = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        epoch += 1

        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.precision.append(logs.get('precision'))
        self.auc.append(logs.get('auc'))
        self.rec.append(logs.get('recall'))

        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_precision.append(logs.get('val_precision'))
        self.val_auc.append(logs.get('val_auc'))
        self.val_rec.append(logs.get('val_recall'))
        self.logs.append(logs)

        def save_plot(*args, title=''):
            N = np.arange(0, len(self.loss))
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            # plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure(figsize=(12.8, 7.2))

            for arg in args:
                plt.plot(N, arg[0], label=arg[1])

            plt.title(title)
            plt.xlabel("Epoch #")
            plt.ylabel("Scalar Value")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('output/{}.png'.format(title))
            plt.close()

        if epoch == self.epochs:
            check_folder('output/')
            save_plot(
                (self.acc, "train_acc"),
                # (self.precision, "train_precision"),
                # (self.auc, "train_auc"),
                # (self.rec, "train_rec"),
                (self.val_acc, "val_accuracy"),
                # (self.val_precision, "val_precision"),
                # (self.val_auc, "val_auc"),
                # (self.val_rec, "val_recall"),
                title="Holdout Scalars [Epoch {}]".format(epoch))

            save_plot((self.loss, "train_loss"),
                      (self.val_loss, "val_loss"),
                      title="Holdout Loss [Epoch {}]".format(epoch))
