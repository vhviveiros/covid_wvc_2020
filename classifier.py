import pandas as pd
import datetime
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, f1_score
from models import classifier_model
from utils import check_folder
from training_plot import TrainingPlot


class Classifier:
    """
    Uso: 
    input_file = your_file.csv when you want to use a new model
    import_model = if you saved a model and want to import it
    """

    def __init__(self, input_file=None, import_model=None):
        if input_file is not None:
            self.read_file(input_file)
        if import_model is not None:
            self.__import_model(import_model)

    def read_file(self, file, test_size=0.2):
        # Read csv
        ctcs = pd.read_csv(file)
        entries = ctcs.iloc[:, 1:269].values
        results = ctcs.iloc[:, 269].values

        # Split into test and training
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            entries, results, test_size=test_size, random_state=0)

        # Normalize data
        sc = StandardScaler()
        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.transform(X_test)

    def __format_validation(self, grid_cv):
        def key_filter(key):
            return list(filter(lambda x: x.startswith('split') and x.__contains__(key), grid_cv.cv_results_))

        return {metric: [grid_cv.cv_results_[m][grid_cv.best_index_] for m in key_filter(metric)]
                for metric in self.metrics}

    @staticmethod
    def custom_specificity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]).ravel()
        return (tn / (tn + fp))

    @staticmethod
    def custom_sensitivity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]).ravel()
        return (tp / (tp + fn))

    def validation(self, n_jobs=-2, cv=10, batch_size=-1, epochs=-1, units=-1, optimizer=['adam'], activation=['relu'], activation_output=['sigmoid'], loss=['binary_crossentropy'], save_path=None):
        classifier = KerasClassifier(build_fn=classifier_model)

        parameters = {'batch_size': batch_size,
                      'epochs': epochs,
                      'units': units,
                      'optimizer': optimizer,
                      'activation': activation,
                      'activationOutput': activation_output,
                      'loss': loss}

        self.metrics = {'accuracy': 'accuracy',
                        'precision': 'precision',
                        'f1_score': make_scorer(f1_score),
                        'sensitivity': make_scorer(Classifier.custom_sensitivity),
                        'specificity': make_scorer(Classifier.custom_specificity)}

        grid_search = GridSearchCV(estimator=classifier,
                                   verbose=2,
                                   param_grid=parameters,
                                   n_jobs=n_jobs,
                                   scoring=self.metrics,
                                   refit='accuracy',
                                   return_train_score=False,
                                   cv=cv)

        grid_search.fit(self.X_train, self.y_train)

        if save_path is not None and len(batch_size) + len(epochs) == 2:
            self.__save_validation(grid_search, save_path)

        return grid_search

    def __save_validation(self, grid_search, save_path):
        result_set = self.__format_validation(grid_search)
        pd.DataFrame(result_set).to_csv(save_path)

    def fit(self, logs_folder, export_dir=None, batch_size=16, epochs=300, units=180, optimizer='sgd', activation='relu', activation_output='sigmoid', loss='binary_crossentropy'):
        date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        check_folder(logs_folder, False)
        log_dir = logs_folder + date_time
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        self.model = classifier_model(optimizer, activation, activation_output, units,
                                      ['accuracy', Precision(), AUC(), Recall()], loss)
        history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=1, workers=12, use_multiprocessing=True,
                                 validation_data=(self.X_test, self.y_test), callbacks=[TrainingPlot(epochs), tensorboard_callback])
        if export_dir is not None:
            self.__export_model(export_dir, date_time)

    def __export_model(self, save_dir, date_time):
        check_folder(save_dir, False)
        self.model.save(save_dir + 'save_' + date_time + '.h5')

    def __import_model(self, model_dir, optimizer='sgd', activation='relu', activation_output='sigmoid', loss='binary_crossentropy', units=180):
        self.model = classifier_model(optimizer, activation, activation_output, units,
                                      ['accuracy', Precision(), AUC(), Recall()], loss)
        self.model.load_weights(model_dir)
        return self.model

    def __confusion_matrix(self):
        pred = self.model.predict_classes(self.X_test)
        matrix = confusion_matrix(pred, self.y_test)
        return(matrix)

    def plot_confusion_matrix(self,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True,
                              save_dir=None):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                    the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                    see http://matplotlib.org/examples/color/colormaps_reference.html
                    plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                    If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                # sklearn.metrics.confusion_matrix
                            normalize    = True,                # show proportions
                            target_names = y_labels_vals,       # list of names of the classes
                            title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        cm = self.__confusion_matrix()
        target_names = ['Covid-19', 'Normal']

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass))
        if save_dir is not None:
            plt.savefig(save_dir)
        plt.show()

    def predict(self, x):
        pred = self.model.predict_classes(x)
        print(pred)
        return pred
