# This work is based on https://github.com/kapil-varshney/utilities/blob/master/training_plot/trainingplot.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.keras
# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
matplotlib.use('agg')


class TrainingPlot(tensorflow.keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.loss = []
        self.val_loss = []
        self.val_acc = []
        self.val_precision = []
        self.val_auc = []
        self.val_rec = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_precision.append(logs.get('val_precision'))
        self.val_auc.append(logs.get('val_auc'))
        self.val_rec.append(logs.get('val_recall'))
        self.logs.append(logs)

        # Before plotting ensure at least 2 epochs have passed
        if len(self.loss) > 1:

            N = np.arange(0, len(self.loss))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            # plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure(figsize=(12.8, 7.2))
            # plt.plot(N, self.loss, label="train_loss")
            # plt.plot(N, self.acc, label="train_acc")
            # plt.plot(N, self.precision, label="train_precision")
            # plt.plot(N, self.auc, label="train_auc")
            # plt.plot(N, self.rec, label="train_rec")
            #plt.plot(N, self.val_loss, label="val_loss")
            plt.plot(N, self.val_acc, label="val_accuracy")
            plt.plot(N, self.val_precision, label="val_precision")
            plt.plot(N, self.val_auc, label="val_auc")
            plt.plot(N, self.val_rec, label="val_recall")

            plt.title(
                "Training Scalars [Epoch {}]".format(epoch + 1))
            plt.xlabel("Epoch #")
            plt.ylabel("Scalar Value")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('output/Epoch-{}.png'.format(epoch + 1))
            plt.close()
