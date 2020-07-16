# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:32 2020

@author: vhviv
"""
# %%Imports
from classifier import Classifier
from utils import abs_path, check_folder
import joblib

# Read data
cf = Classifier(input_file='characteristics.csv')

# %%Model validation
#val = cf.validation(batch_size=[16, 20, 24], epochs=[250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 2000], units=[150, 180, 200, 220, 250, 300, 325], cv=10)
# , 'sgd', 'adadelta'

# val = cf.validation(batch_size=[16], epochs=[300], units=[180, 220],
#                     optimizer=['adadelta'],
#                     activation=['relu', 'elu', 'selu', 'tanh', 'softsign', 'softplus'],
#                     activation_output=['sigmoid', 'softmax', 'tanh', 'softplus'],
#                     loss=['mean_squared_error', 'kl_divergence', 'poisson', 'binary_crossentropy'], cv=10, n_jobs=3)

# val = cf.validation(activation=['relu'], activation_output=['tanh'], batch_size=[16], epochs=[300],
#                     loss=['mean_squared_error'], optimizer=['sgd'], units=[180], cv=10, layers=[7])

# txt = open("result.txt", "a")
# txt.write("###Best_params\n")
# txt.write(str(val.best_params_))
# txt.write("\n\n###Best_index\n")
# txt.write(str(val.best_index_))
# txt.write("\n\n###Best_score\n")
# txt.write(str(val.best_score_))
# txt.close()

# %%Model generate table
# val = cf.validation(activation=['relu'], activation_output=['sigmoid'], batch_size=[16], epochs=[300],
#                     loss=['binary_crossentropy'], optimizer=['sgd'], units=[220], cv=10, n_jobs=-1, save_path='result_table.csv')

# val = cf.validation(batch_size=[16], epochs=[300], units=[180],
#                     optimizer=['sgd'],
#                     activation=['relu'],
#                     activation_output=['tanh'],
#                     loss=['mean_squared_error'],
#                     cv=10, n_jobs=10, save_path='result_table.csv')

# txt = open("result.txt", "a")
# txt.write("\n\n###Best_params\n")
# txt.write(str(val.best_params_))
# txt.write("\n\n###Best_index\n")
# txt.write(str(val.best_index_))
# txt.write("\n\n###Best_score\n")
# txt.write(str(val.best_score_))
# txt.close()

# %%Model train
cf.fit(logs_folder=abs_path("logs\\"),
       export_dir=abs_path('teste/'), epochs=5000)

# %%Read model
#cf = Classifier(import_model=abs_path('teste/save_2020_06_24-17_35_07.h5'))

# %%Results
# Comando para executar Tensorboard
# tensorboard --logdir logs/
# print(history.history.keys())
# %%
