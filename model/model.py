import sklearn
import mne
import numpy as np
import glob
import os
import sys
import logging
import time
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend

from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV

from model.model_file import param_grid_dict

os.environ["PYTHONUNBUFFERED"] = "1"


def split_train_test_path_list(data_path, file_name_template, train_ratio):
    file_list = sorted(glob.glob(os.path.join(data_path, file_name_template)))
    np.random.shuffle(file_list)
    split_id = int(len(file_list) * train_ratio)

    train_list = file_list[:split_id]
    test_list = file_list[split_id:]

    return train_list, test_list


def read_eeg_epochs(train_list, test_list):
    epochs_train_list = []
    epochs_test_list = []

    for file_path in train_list:
        with mne.utils.use_log_level("ERROR"):
            epoch_train = mne.read_epochs(file_path, preload=True)
            epochs_train_list.append(epoch_train)

    for file_path in test_list:
        with mne.utils.use_log_level("ERROR"):
            epoch_test = mne.read_epochs(file_path, preload=True)
            epochs_test_list.append(epoch_test)

    epochs_train = mne.concatenate_epochs(epochs_train_list)
    epochs_test = mne.concatenate_epochs(epochs_test_list)

    return epochs_train, epochs_test


def get_X_and_Y_from_epochs(train_list, test_list, events, picks=None, t_min = -0.2, t_max = 0.5):

    epochs_train, epochs_test = read_eeg_epochs(train_list, test_list)

    epochs_train_list_event1 = epochs_train[events[0]].get_data(picks=picks, tmin=t_min, tmax=t_max)
    epochs_train_list_event2 = epochs_train[events[1]].get_data(picks=picks, tmin=t_min, tmax=t_max)

    labels_up_train = [0] * len(epochs_train_list_event1)
    labels_inv_train = [1] * len(epochs_train_list_event2)

    X_train = np.concatenate((epochs_train_list_event1, epochs_train_list_event2), axis=0)
    y_train = np.concatenate((labels_up_train, labels_inv_train), axis=0)

    epochs_test_list_event1 = epochs_test[events[0]].get_data(picks=picks, tmin=t_min, tmax=t_max)
    epochs_test_list_event2 = epochs_test[events[1]].get_data(picks=picks, tmin=t_min, tmax=t_max)

    labels_up_test = [0] * len(epochs_test_list_event1)
    labels_inv_test = [1] * len(epochs_test_list_event2)

    X_test = np.concatenate((epochs_test_list_event1, epochs_test_list_event2), axis=0)
    y_test = np.concatenate((labels_up_test, labels_inv_test), axis=0)


    return X_train, X_test, y_train, y_test


def train_and_test_model(X_train, X_test, y_train, y_test, pipeline):

    pipeline.fit(X_train, y_train)

    # predict test data
    y_test_pred = pipeline.predict(X_test)
    test_score = accuracy_score(y_test, y_test_pred)

    # predict train data
    y_train_pred = pipeline.predict(X_train)
    train_score = accuracy_score(y_train, y_train_pred)

    print(f"test_score: {test_score:.4f}")
    print(f"train_score: {train_score:.4f}")

def manual_cross_validation(X_train, y_train, model, k=3):
    folds = np.array_split(np.arange(len(X_train)), k)

    scores = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
        y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_test_fold)

        accuracy = accuracy_score(y_test_fold, y_pred)
        scores.append(accuracy)

    mean_accuracy = np.mean(scores)
    return mean_accuracy

# To make smaller data sets to tests manual grid search
def medium_data_set():
    X, y = make_classification(
        n_samples=5000,
        n_features=100,
        n_informative=75,
        n_redundant=0,
        n_classes=2,
        class_sep=1.5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # to maintain class proportion
    )
    return X_train, X_test, y_train, y_test

def make_small_data_set():
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


#----- LOGGING -----

log_file = "training_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"), # a to overwrite
        logging.StreamHandler()
    ]
)

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.line_buffer = ""

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)



if __name__ == "__main__":
    dir_path = 'D:\studia\magisterka\dane EEG\BADANIE_POLITYCZNE_2022_eeg_bdfy\EEG_preprocessed'
    file_name_template = "s*.bdf-epo.fif"
    train_ratio = 0.8

    flatten_transformer = FunctionTransformer(lambda X: X.reshape(X.shape[0], -1))

    param_grid = dict(
        svc__kernel='linear',
        svc__C=0.01,
        svc__gamma=['scale', 'auto', 0.001, 0.01, 0.1, 1],
    )

    train_list, test_list = split_train_test_path_list(dir_path, file_name_template, train_ratio)
    X_train, X_test, y_train, y_test = get_X_and_Y_from_epochs(train_list, test_list, ["up", "inv"], t_min=0.0,
                                                               t_max=0.25)

    logging.info("Rozpoczynam trenowanie modeli")

    for gamma in param_grid['svc__gamma']:
        model = Pipeline(steps=[
            ('reshape', flatten_transformer),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel=param_grid['svc__kernel'], C=param_grid['svc__C'], gamma=gamma))
        ])

        logging.info(f"Trenuję model z gamma={gamma}")
        logging.info(f"Trenuję model z C=1")
        logging.info(f"Trenuję model z kernel='linear'")

        mean_accuracy = manual_cross_validation(X_train, y_train, model, k=5)

        logging.info(f"Średnia dokładność dla gamma={gamma}: {mean_accuracy}")

    logging.info("Trenowanie zakończone.")

