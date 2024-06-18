import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import './lab11.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
import librosa
import IPython.display as ipd
import librosa.display
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
#%matplotlib notebook
# %matplotlib inline

# Enter the data path
PATH_DATASET = "/kaggle/input/dataset"

"""Function to create dataframes for all machine types containing the filename of each sample, its section, its attribute and type of sound."""

def build_dataframe(machine_str = 'valve'):
    #Get list of files in train and test directory
    path_train_folder = PATH_DATASET + "/dev_" + machine_str + "/" + machine_str + "/train"
    path_test_folder = PATH_DATASET  + "/dev_" + machine_str + "/" + machine_str + "/test"

    train_files = [f for f in os.listdir(path_train_folder)]
    test_files = [f for f in os.listdir(path_test_folder) ]
    #Get list of dictionnary for creating DataFrame
    list_dict_file = []

    #Loop through filenames
    for filename in train_files:

        #Get filename as list of string
        splitted_filename = filename.split('_')

        #Append dictionnary to list
        list_dict_file.append({
            'filepath' : path_train_folder + "/" + filename,
            'filename' : filename,
            # Handle non-integer values in 'section' column
            'section' : splitted_filename[1] if splitted_filename[1].isdigit() else None,
            'domain_env' : splitted_filename[2],
            'dir' : splitted_filename[3],
            'sound_type' : splitted_filename[4],
            'id' : splitted_filename[5],
            'suffix' : '_'.join(splitted_filename[6:]).split('.wav')[0]
        })

    #Loop through filenames
    for filename in test_files:

        #Get filename as list of string
        splitted_filename = filename.split('_')
        #Append dictionnary to list
        list_dict_file.append({
            'filepath' : path_test_folder  + "/" + filename,
            'filename' : filename,
            # Handle non-integer values in 'section' column
            'section' : splitted_filename[1] if splitted_filename[1].isdigit() else None,
            'domain_env' : splitted_filename[2],
            'dir' : splitted_filename[3],
            'sound_type' : splitted_filename[4],
            'id' : splitted_filename[5],
            'suffix' : '_'.join(splitted_filename[6:]).split('.wav')[0]
        })

    return pd.DataFrame(list_dict_file)

"""Function to get the data matrix of amplitude spectrograms and the data matrix of phase spectrograms from a dataframe"""

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr = None)
    return y, sr

def spectrogram(audio, n_fft = 1024, hop_length = 512):
    spectrum = librosa.stft(audio, n_fft = n_fft, hop_length = hop_length, center = False)
    # Chose center = False above after doing a test on a simple superposition of two sin with two frequencies
    # There are boundary artifacts in the spectrograms if center = True
    # Not investigated why it is like this but the spectrograms look definitely better with center = False
    # To be confirmed
    magnitude, phase = librosa.magphase(spectrum)
    magnitude_in_db = librosa.amplitude_to_db(magnitude, ref=1e-6)
    # or ref=np.max. ref = 1e-6 corresponds to the threshold intensity for humans = 1e-12 W/m2
    # Not sure of the units. Does not matter, I just want a fixed ref for all spectrograms
    return magnitude_in_db, np.angle(phase)

def get_spectros_from_df(df, n_fft = 1024, hop_length = 512):
    def path_to_spectra(path):
        y, sr = load_audio(path)
        mag_sp, phase_sp = spectrogram(y, n_fft = n_fft, hop_length = hop_length)
        return mag_sp.flatten(), phase_sp.flatten()

    filepaths = df['filepath'].reset_index(drop = True)
    X_tmp = path_to_spectra(filepaths.iloc[0])[0]     # to get the size of a flatten spectrum
    X_mag = np.empty((filepaths.shape[0], X_tmp.size))
    X_phase = np.empty((filepaths.shape[0], X_tmp.size))

    for i, path in filepaths.items():
        mag_sp, phase_sp = path_to_spectra(path)
        X_mag[i] = mag_sp
        X_phase[i] = phase_sp

    return X_mag, X_phase


#X_mag, X_phase = get_spectros_from_df(df_gearbox[(df_gearbox['dir']=='test') & (df_gearbox['section']==0)])


# Choose the machine
machine_str = 'gearbox'

# Fix the parameters
params = dict(n_fft = 1024,        # n_fft paramater for calculating the spectrograms with librosa.stft
              hop_length = 512     # hop_length paramater for calculating the spectrograms with librosa.stft
             )

df = build_dataframe(machine_str)

df_normal = df[df['sound_type']=='normal'].sample(n = 50, random_state = 1)
df_anormal = df[df['sound_type']=='anomaly'].sample(n = 50, random_state = 1)
data = pd.concat([df_normal, df_anormal], axis = 0).reset_index()

X_mag, X_phase = get_spectros_from_df(data, n_fft = params['n_fft'], hop_length = params['hop_length'])

target = data['sound_type']
target = target.replace(to_replace = ['normal', 'anomaly'], value = [0, 1])

# Choose here if the training is done on the amplitude spectrogram, phase spectrogram or both
which = 'amplitude'      # 'amplitude', 'phase', or 'both'

def my_train_test_split(X_mag, X_phase, target, which = 'amplitude', test_size = 0.2, random_state = 123):
    if which == 'amplitude':
        X = X_mag
    elif which == 'phase':
        X = X_phase
    elif which == 'both':
        X = np.concatenate([X_mag, X_phase], axis = 1)
    else:
        raise ValueError("'which' must be equal to 'amplitude', 'phase' or 'both'")

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = my_train_test_split(X_mag, X_phase, target, which = which)

X_red_train, X_red_test = X_train, X_test


# We reduce the number of features by computing the means of the spectrograms per column
def dim_reduce_by_spcol(X_train, X_test, n_fft = 1024, \
                        mean = True, median = False, mini = False, maxi = False, std = False):
    # Reshape 2D X_train, X_test to 3D arrays ie unflatten the spectrograms
    # The number of lines in the original spectrogram is 1 + int(n_fft/2)
    X_train = np.reshape(X_train, (X_train.shape[0], 1 + int(n_fft/2), -1))
    X_test = np.reshape(X_test, (X_test.shape[0], 1 + int(n_fft/2), -1))

    X_train_red = np.empty((X_train.shape[0], 0))
    X_test_red = np.empty((X_test.shape[0], 0))
    if mean:
        X_train_red  = np.hstack([X_train_red, np.mean(X_train, axis = 1)])
        X_test_red  = np.hstack([X_test_red, np.mean(X_test, axis = 1)])
    if median:
        X_train_red  = np.hstack([X_train_red, np.median(X_train, axis = 1)])
        X_test_red  = np.hstack([X_test_red, np.median(X_test, axis = 1)])
    if mini:
        X_train_red  = np.hstack([X_train_red, np.amin(X_train, axis = 1)])
        X_test_red  = np.hstack([X_test_red, np.amin(X_test, axis = 1)])
    if maxi:
        X_train_red  = np.hstack([X_train_red, np.amax(X_train, axis = 1)])
        X_test_red  = np.hstack([X_test_red, np.amax(X_test, axis = 1)])
    if std:
        X_train_red  = np.hstack([X_train_red, np.std(X_train, axis = 1)])
        X_test_red  = np.hstack([X_test_red, np.std(X_test, axis = 1)])

    return X_train_red, X_test_red

# Choose how to reduce the spectrograms
params_red = dict(mean = True,
                  median = False,
                  mini = False,
                  maxi = False,
                  std = True
                 )

X_red_train, X_red_test = dim_reduce_by_spcol(X_train, X_test, n_fft = params['n_fft'], **params_red)

print(X_red_train.shape)
print(X_train.shape)

sel = SelectPercentile(score_func = f_classif, percentile = 30)
sel.fit(X_train, y_train)
X_red_train = sel.transform(X_train)
X_red_test = sel.transform(X_test)

n_feats = X_red_train.shape[1]
pca = PCA(n_components = 0.85)       # keep 85% of variance
pca.fit(X_red_train)
X_red_train = pca.transform(X_red_train)
X_red_test = pca.transform(X_red_test)

print("Number of initial features = ", n_feats)
print("Number of selected features = ", pca.n_components_)

"""I find empirically that n_components = 0.85 is a good choice (looking at test AUC calculated with xgboost) but this should be done more rigorously"""

plt.scatter(X_red_train[:,0], X_red_train[:,1], c = y_train)
plt.xlabel("PCA axis 0")
plt.ylabel("PCA axis 1")
plt.title("Projection of the first two PCA axis");

"""## Machine learning

#### First classification method : Gradient Boosting
"""

# Convert arrays to DMatrices
M_red_train = xgb.DMatrix(X_red_train, y_train)
M_red_test = xgb.DMatrix(X_red_test, y_test)

# Training
params_xgb = {'booster': 'gbtree',
              'learning_rate': 0.3,
              'alpha': 0.001,     # L1 regularization term
              'eval_metric': 'error',
              'objective': 'binary:logistic'}

xgb_model = xgb.train(dtrain = M_red_train, params = params_xgb, num_boost_round = 1000, \
                      early_stopping_rounds = 20, evals = [(M_red_train, 'train'), (M_red_test, 'test')])

train_pred_probas = xgb_model.predict(M_red_train, iteration_range = (0, xgb_model.best_iteration + 1))
train_preds = pd.Series(np.where(train_pred_probas>0.5, 1, 0))

test_pred_probas = xgb_model.predict(M_red_test, iteration_range = (0, xgb_model.best_iteration + 1))
test_preds = pd.Series(np.where(test_pred_probas>0.5, 1, 0))


ratios = np.linspace(0, 1, 10)

train_errors = []    # list to store train errors
test_errors = []    # list to store test errors
for ratio in ratios:
    params_xgb = {'booster': 'gbtree',
          'learning_rate': 1,
          'alpha': 0.001,     # L1 regularization term
          'subsample': ratio,
          'eval_metric': 'error',
          'objective': 'binary:logistic'}

    xgb_model = xgb.train(dtrain = M_red_train, params = params_xgb, num_boost_round = 1000, \
                  early_stopping_rounds = 20, evals = [(M_red_train, 'train'), (M_red_test, 'test')], \
                  verbose_eval = False)

    train_pred_probas = xgb_model.predict(M_red_train, iteration_range = (0, xgb_model.best_iteration + 1))
    train_preds = pd.Series(np.where(train_pred_probas>0.5, 1, 0))
    train_error = 1 - accuracy_score(y_train, train_preds)

    test_pred_probas = xgb_model.predict(M_red_test, iteration_range = (0, xgb_model.best_iteration + 1))
    test_preds = pd.Series(np.where(test_pred_probas>0.5, 1, 0))
    test_error = 1 - accuracy_score(y_test, test_preds)

    train_errors.append(train_error)
    test_errors.append(test_error)

fig = plt.figure(figsize = (7, 5))
plt.plot(ratios, train_errors, 'r-o', label = 'Train error')
plt.plot(ratios, test_errors, 'b-s', label = 'Test error')
plt.xlabel("Fraction of sampled observations")
plt.ylabel("Train/test errors")
plt.legend()

learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
alphas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

aucs = []    # list to store auc
for alpha in alphas:
    for learning_rate in learning_rates:
        params_xgb = {'booster': 'gbtree',
              'learning_rate': learning_rate,
              'alpha': alpha,     # L1 regularization term
              'eval_metric': 'error',
              'objective': 'binary:logistic'}

        xgb_model = xgb.train(dtrain = M_red_train, params = params_xgb, num_boost_round = 1000, \
                      early_stopping_rounds = 20, evals = [(M_red_train, 'train'), (M_red_test, 'test')], \
                      verbose_eval = False)

        pred_probas = xgb_model.predict(M_red_test, iteration_range = (0, xgb_model.best_iteration + 1))
        preds = pd.Series(np.where(pred_probas>0.5, 1, 0))
        auc = roc_auc_score(y_test, preds)
        aucs.append([alpha, learning_rate, auc])

aucs = np.array(aucs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
for lr in learning_rates:
    aucs_lr = aucs[aucs[:,1]==lr]
    ax1.plot(aucs_lr[:,0], aucs_lr[:,2], "-o", label = "lr = " + str(lr))
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("AUC")
    ax1.set_xscale('log')
    ax1.legend(loc = 'best')

for alpha in alphas:
    aucs_alpha = aucs[aucs[:,0]==alpha]
    ax2.plot(aucs_alpha[:,1], aucs_alpha[:,2], "-o", label = "alpha = " + str(alpha))
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("AUC")
    ax2.set_xscale('log')
    ax2.legend(loc = 'best')

test_f1_score_0 = f1_score(y_test, test_preds, pos_label = 0)
test_f1_score_1 = f1_score(y_test, test_preds, pos_label = 1)
train_f1_score_0 = f1_score(y_train, train_preds, pos_label = 0)
train_f1_score_1 = f1_score(y_train, train_preds, pos_label = 1)
test_auc = roc_auc_score(y_test, test_preds)
train_auc = roc_auc_score(y_train, train_preds)

print("f1 score of class 0 on test set = ", np.round(test_f1_score_0, 3))
print("f1 score of class 1 on test set = ", np.round(test_f1_score_1, 3))
print("f1 score of class 0 on train set = ", np.round(train_f1_score_0, 3))
print("f1 score of class 1 on train set = ", np.round(train_f1_score_1, 3))
print("Train AUC = ", np.round(train_auc, 3))
print("Test AUC = ", np.round(test_auc, 3))

#pd.crosstab(y_test.values, test_preds.values, rownames = ['Classe réelle'], colnames = ['Classe prédite'])
pd.crosstab(y_test, test_preds, rownames = ['Classe réelle'], colnames = ['Classe prédite'])


col_names = ["machine",
             "section",
             "which_spectro",
             "Reduction_method",
             "Reduction_params",
             "Classif_method",
             "Classif_params",
             "Train AUC",
             "Test AUC"]

results = pd.DataFrame(columns = col_names)

# 1st try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "phase",
        "Reduction_method": "spectro cols",
        "Reduction_params": 'mean',
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.48
        }

results.loc[len(results)] = line

# 2nd try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": "spectro cols",
        "Reduction_params": 'mean',
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 0.999,
        "Test AUC": 0.608
        }

results.loc[len(results)] = line

# 3rd try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "both",
        "Reduction_method": "spectro cols",
        "Reduction_params": 'mean',
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train score": 1,
        "Train AUC": 1,
        "Test AUC": 0.64
        }

results.loc[len(results)] = line

# 4th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": "spectro cols",
        "Reduction_params": 'mean+med',
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.687
        }

results.loc[len(results)] = line

# 5th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": "spectro cols",
        "Reduction_params": "mean+med+min+max",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.585
        }

results.loc[len(results)] = line

# 6th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "phase",
        "Reduction_method": "spectro cols",
        "Reduction_params": "mean+med+min+max",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.543
        }

results.loc[len(results)] = line

# 7th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "phase",
        "Reduction_method": "spectro cols",
        "Reduction_params": "mean+med",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.482
        }

results.loc[len(results)] = line

# 7th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": "spectro cols",
        "Reduction_params": "mean+med+std",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.685
        }

results.loc[len(results)] = line

# 8th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": "spectro cols",
        "Reduction_params": "mean+std",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.672
        }

results.loc[len(results)] = line

# 9th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "both",
        "Reduction_method": "spectro cols",
        "Reduction_params": "mean+std",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 0.917,
        "Test AUC": 0.588
        }

results.loc[len(results)] = line

# 10th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": None,
        "Reduction_params": None,
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.698
        }

results.loc[len(results)] = line

# 11th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "both",
        "Reduction_method": None,
        "Reduction_params": None,
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.657
        }

results.loc[len(results)] = line

# 12th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "phase",
        "Reduction_method": None,
        "Reduction_params": None,
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.565
        }

results.loc[len(results)] = line

# 13th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile',
        "Reduction_params": "20%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.718
        }

results.loc[len(results)] = line

# 14th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile',
        "Reduction_params": "30%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.767
        }

results.loc[len(results)] = line

# 15th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile',
        "Reduction_params": "40%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.677
        }

results.loc[len(results)] = line

# 16th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "both",
        "Reduction_method": 'Percentile',
        "Reduction_params": "30%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.747
        }

results.loc[len(results)] = line

# 17th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 95%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.788
        }

results.loc[len(results)] = line

# 18th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 90%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.847
        }

results.loc[len(results)] = line

# 19th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 80%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 0.857,
        "Test AUC": 0.767
        }

results.loc[len(results)] = line

# 19th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 85%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.86
        }

results.loc[len(results)] = line

# 20th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "both",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 85%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=1',
        "Train AUC": 1,
        "Test AUC": 0.887
        }

results.loc[len(results)] = line

# 21th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "amplitude",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 85%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.01, lr=0.3',
        "Train AUC": 1,
        "Test AUC": 0.877
        }

results.loc[len(results)] = line

# 22th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "both",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 85%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=0.3',
        "Train AUC": 1,
        "Test AUC": 0.903
        }

results.loc[len(results)] = line

# 22th try ********************
line = {
        "machine": 'gearbox',
        "section": 'all',
        "which_spectro": "phase",
        "Reduction_method": 'Percentile + PCA',
        "Reduction_params": "perc : 30%; pca : 85%",
        "Classif_method": 'xgboost',
        "Classif_params": 'num_boost_r=1000, alpha=0.001, lr=0.3',
        "Train AUC": 1,
        "Test AUC": 0.5
        }

results.loc[len(results)] = line



results.loc[len(results)] = line


# Show the results
results.sort_values(by = 'Test AUC', ascending = False)

def compute_aucs_cv(machine_str, which, params, params_xgb):
    # Create features data and target
    df = build_dataframe(machine_str)

    df_normal = df[df['sound_type']=='normal'].sample(n = 100, random_state = 1)
    df_anormal = df[df['sound_type']=='anomaly'].sample(n = 100, random_state = 1)
    data = pd.concat([df_normal, df_anormal], axis = 0).reset_index()

    X_mag, X_phase = get_spectros_from_df(data, n_fft = params['n_fft'], hop_length = params['hop_length'])

    target = data['sound_type']
    target = target.replace(to_replace = ['normal', 'anomaly'], value = [0, 1])

    # Here I don't want to split in train and test splits and want to keep all the data for the cross-validation
    X_train, X_test, y_train, y_test = my_train_test_split(X_mag, X_phase, target, which = which, test_size = 0.01)

    # Reduction of dimensionality
    print("Number of initial features = ", X_train.shape[1])

    sel = SelectPercentile(score_func = f_classif, percentile = 30)
    sel.fit(X_train, y_train)
    X_red_train = sel.transform(X_train)
    #X_red_test = sel.transform(X_test)

    pca = PCA(n_components = 0.85)       # keep 85% of variance
    pca.fit(X_red_train)
    X_red_train = pca.transform(X_red_train)
    #X_red_test = pca.transform(X_red_test)

    print("Number of selected features after SelectPercentile + PCA = ", pca.n_components_)

    # Convert arrays to DMatrices
    M_red_train = xgb.DMatrix(X_red_train, y_train)
    #M_red_test = xgb.DMatrix(X_red_test, y_test)

    # Cross-val
    seeds = np.arange(50)
    test_aucs = []
    test_errors = []

    for seed in seeds:
        results_cv = xgb.cv(dtrain = M_red_train, params = params_xgb, num_boost_round = 1000, seed = seed, \
                            early_stopping_rounds = 20, metrics = ['auc', 'error'], nfold = 5)

        #display(results_cv)

        test_aucs.append([seed, results_cv['test-auc-mean'].max(), \
                          results_cv['test-auc-std'].iloc[results_cv['test-auc-mean'].argmax()]])
        test_errors.append([seed, results_cv['test-error-mean'].min(), \
                          results_cv['test-error-std'].iloc[results_cv['test-error-mean'].argmin()]])

    return np.array(test_aucs), np.array(test_errors)

# Choose the machine
machine_str = 'gearbox'

# Choose here if the training is done on the amplitude spectrogram, phase spectrogram or both
which = 'amplitude'      # 'amplitude', 'phase', or 'both'

# Fix the parameters
params = dict(n_fft = 1024,        # n_fft paramater for calculating the spectrograms with librosa.stft
              hop_length = 512     # hop_length paramater for calculating the spectrograms with librosa.stft
             )

params_xgb = {'booster': 'gbtree',
              'learning_rate': 0.3,
              'alpha': 0.01,     # L1 regularization term
              'eval_metric': 'error',
              'objective': 'binary:logistic'}

test_aucs_ampl, test_errors_ampl = compute_aucs_cv(machine_str, 'amplitude', params, params_xgb)
# Here I should change the optimized params_xgb for both (not done ...)
test_aucs_both, test_errors_both = compute_aucs_cv(machine_str, 'both', params, params_xgb)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))

ax1.plot(test_aucs_ampl[:, 0], test_aucs_ampl[:, 1], 'r-', label = 'amplitude')
ax1.fill_between(test_aucs_ampl[:, 0], test_aucs_ampl[:, 1] - test_aucs_ampl[:, 2], \
                 test_aucs_ampl[:, 1] + test_aucs_ampl[:, 2], color='r', alpha=0.1)

ax1.plot(test_aucs_both[:, 0], test_aucs_both[:, 1], 'b-', label = 'both')
ax1.fill_between(test_aucs_both[:, 0], test_aucs_both[:, 1] - test_aucs_both[:, 2], \
                 test_aucs_both[:, 1] + test_aucs_both[:, 2], color='b', alpha=0.1)
ax1.set_xlabel("Cross-validation seed")
ax1.set_ylabel("Test AUC")
ax1.set_ylim([0.7, 1])
ax1.legend()
ax1.set_title('gearbox')
ax1.grid(True);

ax2.plot(test_errors_ampl[:, 0], test_errors_ampl[:, 1], 'r-', label = 'amplitude')
ax2.fill_between(test_errors_ampl[:, 0], test_errors_ampl[:, 1] - test_errors_ampl[:, 2], \
                 test_errors_ampl[:, 1] + test_errors_ampl[:, 2], color='r', alpha=0.1)

ax2.plot(test_errors_both[:, 0], test_errors_both[:, 1], 'b-', label = 'both')
ax2.fill_between(test_errors_both[:, 0], test_errors_both[:, 1] - test_errors_both[:, 2], \
                 test_errors_both[:, 1] + test_errors_both[:, 2], color='b', alpha=0.1)
ax2.set_xlabel("Cross-validation seed")
ax2.set_ylabel("Test binary classification error rate")
ax2.set_ylim([0, 0.3])
ax2.legend()
ax2.set_title('gearbox')
ax2.grid(True);


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))

ax1.plot(test_aucs_ampl[:, 0], test_aucs_ampl[:, 1], 'r-', label = 'amplitude')
ax1.fill_between(test_aucs_ampl[:, 0], test_aucs_ampl[:, 1] - test_aucs_ampl[:, 2], \
                 test_aucs_ampl[:, 1] + test_aucs_ampl[:, 2], color='r', alpha=0.1)

ax1.plot(test_aucs_both[:, 0], test_aucs_both[:, 1], 'b-', label = 'both')
ax1.fill_between(test_aucs_both[:, 0], test_aucs_both[:, 1] - test_aucs_both[:, 2], \
                 test_aucs_both[:, 1] + test_aucs_both[:, 2], color='b', alpha=0.1)
ax1.set_xlabel("Cross-validation seed")
ax1.set_ylabel("Test AUC")
ax1.set_ylim([0.7, 1])
ax1.legend()
ax1.set_title('fan')
ax1.grid(True);

ax2.plot(test_errors_ampl[:, 0], test_errors_ampl[:, 1], 'r-', label = 'amplitude')
ax2.fill_between(test_errors_ampl[:, 0], test_errors_ampl[:, 1] - test_errors_ampl[:, 2], \
                 test_errors_ampl[:, 1] + test_errors_ampl[:, 2], color='r', alpha=0.1)

ax2.plot(test_errors_both[:, 0], test_errors_both[:, 1], 'b-', label = 'both')
ax2.fill_between(test_errors_both[:, 0], test_errors_both[:, 1] - test_errors_both[:, 2], \
                 test_errors_both[:, 1] + test_errors_both[:, 2], color='b', alpha=0.1)
ax2.set_xlabel("Cross-validation seed")
ax2.set_ylabel("Test binary classification error rate")
ax2.set_ylim([0, 0.3])
ax2.legend()
ax2.set_title('fan')
ax2.grid(True);

# Fix the parameters
params = dict(n_fft = 1024,        # n_fft paramater for calculating the spectrograms with librosa.stft
              hop_length = 512     # hop_length paramater for calculating the spectrograms with librosa.stft
             )

params_xgb = {'booster': 'gbtree',
              'learning_rate': 0.3,
              'alpha': 0.01,     # L1 regularization term
              'eval_metric': 'error',
              'objective': 'binary:logistic'}

dict_results = {}
for machine in ['gearbox']:
    test_aucs_ampl, test_errors_ampl = compute_aucs_cv(machine, 'amplitude', params, params_xgb)
    dict_results[machine] = test_aucs_ampl, test_errors_ampl

# Save the results into a file
np.save('dict_results_classif_PCAxgboost.npy', dict_results)

# To load the results
#new_dict = np.load('dict_results_classif_PCAxgboost.npy', allow_pickle='TRUE')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

for machine in [ 'gearbox']:
    test_aucs_ampl, test_errors_ampl = dict_results[machine]
    ax1.plot(test_aucs_ampl[:, 0], test_aucs_ampl[:, 1], label = machine)
    ax1.fill_between(test_aucs_ampl[:, 0], test_aucs_ampl[:, 1] - test_aucs_ampl[:, 2], \
                     test_aucs_ampl[:, 1] + test_aucs_ampl[:, 2], alpha=0.1)

    ax2.plot(test_errors_ampl[:, 0], test_errors_ampl[:, 1], label = machine)
    ax2.fill_between(test_errors_ampl[:, 0], test_errors_ampl[:, 1] - test_errors_ampl[:, 2], \
                     test_errors_ampl[:, 1] + test_errors_ampl[:, 2], alpha=0.1)

ax1.set_xlabel("Cross-validation seed", fontsize = 13.0)
ax1.set_ylabel("Test AUC", fontsize = 13.0)
ax1.set_ylim([0.75, 1])
ax1.legend(loc = 'lower right')
ax1.grid(axis = 'y')
ax1.set_title("Supervised classification with PCA + xgboost", fontsize = 13.0)

ax2.set_xlabel("Cross-validation seed", fontsize = 13.0)
ax2.set_ylabel("Test binary classification error rate", fontsize = 13.0)
#ax2.set_ylim([0, 0.3])
ax2.legend(loc = 'lower right')
ax2.grid(axis = 'y')
ax2.set_title("Supervised classification with PCA + xgboost", fontsize = 13.0);

dict_aucs = {}
dict_errors = {}

for machine in [ 'gearbox']:
    test_aucs_ampl, test_errors_ampl = dict_results[machine]
    dict_aucs[machine] = test_aucs_ampl[:,1].mean()
    dict_errors[machine] = test_errors_ampl[:,1].mean()

print(dict_aucs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 3))

ax1.bar(range(1), dict_aucs.values())
ax1.set_xticks(range(1))
ax1.set_xticklabels(dict_aucs.keys(), fontsize = 13.0)
ax1.set_ylabel("Mean test AUC", fontsize = 13.0)
ax1.grid(True, axis = 'y')

ax2.bar(range(1), dict_errors.values())
ax2.set_xticks(range(1))
ax2.set_xticklabels(dict_errors.keys(), fontsize = 13.0)
ax2.set_ylabel("Mean test binary \n classification error rate", fontsize = 13.0)
ax2.grid(True, axis = 'y')

dict_results = np.load('dict_results_classif_PCAxgboost.npy', allow_pickle='TRUE')

test_aucs_ampl, test_errors_ampl = dict_results[()]['gearbox']
print("AUC = ", test_aucs_ampl[:,1].mean())
print('Error =', test_errors_ampl[:,1].mean())
`;

const codeSections = {
  Step1: `
# Install libraries if not already installed


!pip install tensorflow scikit-learn


# Import libraries import numpy as np import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
 
from tensorflow.keras.applications import VGG16 from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.metrics.pairwise import cosine_similarity from sklearn.neighbors import KNeighborsClassifier from sklearn.cluster import KMeans
from sklearn.metrics import classification_report import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = '/kaggle/input/real-life-industrial-dataset-of-casting-product'


# Load dataset (assuming data is organized in directories by class) datagen = ImageDataGenerator(rescale=1./255)
dataset = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary')


# Function to load and prepare the data def load_data(dataset):
features = [] labels = []
for batch in dataset:
X_batch, y_batch = batch features.extend(X_batch) labels.extend(y_batch)
if len(features) >= dataset.samples: break
return np.array(features), np.array(labels)


X, y = load_data(dataset)
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Normalize and reshape the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 224 * 224 * 3))
X_test = scaler.transform(X_test.reshape(-1, 224 * 224 * 3))


# Load pre-trained VGG16 model for feature extraction
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


# Function to extract features
def extract_features(model, data): features = model.predict(data)
return features.reshape(features.shape[0], -1)


X_train_features = extract_features(model, X_train.reshape(-1, 224, 224, 3))
X_test_features = extract_features(model, X_test.reshape(-1, 224, 224, 3))

`,
  SplitData: `
from sklearn.model_selection import train_test_split

x_train , x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

import matplotlib.pyplot as plt 

plt.imshow(x_train[1])
print(y_train[1])
`,
  DeepLearningModel: `
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  
])

model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Use SparseCategoricalCrossentropy for multi-class classification
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
`,

  TrainModel: `
  model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(x_val, y_val))

  `
};

const Lab2 = () => {
  const [highlightedCodeSnippet, setHighlightedCodeSnippet] = useState("");


  useEffect(() => {

    hljs.highlightAll();
  }
  )

  const ParticleCanvas = () => {
    const canvasRef = useRef(null);
  
    useEffect(() => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      let particles = [];
  
      // Function to create a particle
      function Particle(x, y) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 0.4 * 5 + 0.85; // 15% of the size
        this.speedX = Math.random() * 3 - 1.5;
        this.speedY = Math.random() * 3 - 1.5;
      }
  
      // Function to draw particles and connect them with lines
      function drawParticles() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
  
        for (let i = 0; i < particles.length; i++) {
          ctx.fillStyle = 'orangered'; // Change particle color to orangered
          ctx.beginPath();
          ctx.arc(particles[i].x, particles[i].y, particles[i].size, 0, Math.PI * 2);
          ctx.fill();
  
          particles[i].x += particles[i].speedX;
          particles[i].y += particles[i].speedY;
  
          // Wrap particles around the screen
          if (particles[i].x > canvas.width) particles[i].x = 0;
          if (particles[i].x < 0) particles[i].x = canvas.width;
          if (particles[i].y > canvas.height) particles[i].y = 0;
          if (particles[i].y < 0) particles[i].y = canvas.height;
  
          // Draw lines between neighboring particles
          for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const opacity = 1 - distance / 100; // Opacity based on distance
  
            if (opacity > 0) {
              ctx.strokeStyle = `rgba(0, 0, 0, ${opacity})`; // Set line opacity
              ctx.lineWidth = 0.5; // Set line thickness
              ctx.beginPath();
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.stroke();
            }
          }
        }
  
        requestAnimationFrame(drawParticles);
      }
  
      for (let i = 0; i < 120; i++) {
        particles.push(new Particle(Math.random() * canvas.width, Math.random() * canvas.height));
      }
  
      drawParticles();
  
      return () => {
        particles = [];
      };
    }, []);
  
    return <canvas ref={canvasRef} style={{ position: 'fixed', zIndex: -1, top: 0, left: 0, width: '100vw', height: '100vh' }} />;
  };

  const handleHeadingClick = (section) => {
    const snippet = codeSections[section];
    setHighlightedCodeSnippet(snippet);
  };

  return (
    <div className="dashboard">
      <ParticleCanvas />
      <div className="Layout" style={{ display: "flex", justifyContent: "space-around", color: '#09F' }}>
      <div className="box3">
      <h2>XGBoost: Extreme Gradient Boosting Explained</h2> <br />

      <p>XGBoost, short for "Extreme Gradient Boosting," is a scalable and highly efficient machine 
        learning algorithm used for supervised learning tasks, including classification and regression. 
        Developed by Tianqi Chen, XGBoost has become one of the most popular algorithms in the field 
        due to its speed, performance, and versatility. Here's a detailed look into XGBoost, supported by 
        theoretical explanations and illustrative images.</p> <br />

      <h3>Theory Behind XGBoost</h3> <br />
      <ul>
        <li><strong>Gradient Boosting Framework:</strong> XGBoost is based on the gradient boosting framework, 
          which builds models sequentially. Each new model attempts to correct the errors of the 
          previous models. The overall model is a weighted sum of all previous models.</li> <br />
        <li><strong>Decision Trees:</strong> XGBoost primarily uses decision trees as its base learners. These are 
          simple models that split the data into branches based on feature values to make 
          predictions.</li> <br />
        <li><strong>Boosting:</strong> Boosting is an ensemble technique that combines the outputs of several weak 
          learners to create a strong learner. Each new model is trained to correct the errors made 
          by the previous models, focusing more on the difficult-to-predict instances.</li> <br />
        <li><strong>Gradient Descent:</strong> In XGBoost, the new models are fit on the negative gradient of the 
          loss function. This means each new model aims to reduce the errors (residuals) of the 
          previous models by moving in the direction of the steepest descent (gradient) of the loss 
          function.</li> <br />
      </ul> <br />
      <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />

      <h3>Mathematical Formulation and Visual Illustration</h3> <br />
      <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />
      <p><strong>Gradient Boosting Process:</strong></p>
      <ol>
        <li><strong>Step 1:</strong> Start with an initial prediction (e.g., the mean of the target values).</li> <br />
        <li><strong>Step 2:</strong> Compute the residuals (errors) between the actual and predicted values.</li> <br />
        <li><strong>Step 3:</strong> Fit a new model to the residuals.</li> <br />
        <li><strong>Step 4:</strong> Update the predictions by adding the new model's predictions, scaled by a 
          learning rate.</li> <br />
        <li><strong>Step 5:</strong> Repeat steps 2-4 for a specified number of iterations or until the residuals are 
          minimized.</li> <br />
      </ol>

      <p><strong>Decision Tree Example:</strong></p> <br />
      <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />
      <p>XGBoost uses decision trees as base learners. Each tree splits the data based on 
        feature values to make predictions.</p> <br />
        <img style={{width: '100%'}} src={Img4} alt="image4" /> <br /> <br />
    </div>

        <div className="box4">
          <div className="code-container">
            <pre className="code-snippet">
              <code className="python" >
              {highlightedCodeSnippet ? highlightedCodeSnippet.trim() : codeSnippet2.trim()}
              </code>
            </pre>
          </div>
        </div>
      </div>
      <div> 
          <button className="button">
          <a href="https://www.kaggle.com/code/priyanshsurana/notebook7ea446b9bf?scriptVersionId=183970208" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
