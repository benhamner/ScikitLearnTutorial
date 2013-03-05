import csv
import json
import numpy as np
import os
import pickle

def get_paths():
    paths = json.loads(open("Settings.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def get_train():
    paths = get_paths()
    train = np.loadtxt(open(paths["train_data_path"]), delimiter=",")
    y_train = np.loadtxt(open(paths["train_labels_path"]), delimiter=",")
    return train, y_train

def get_test():
    paths = get_paths()
    return np.loadtxt(open(paths["test_data_path"]), delimiter=",")

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def write_submission(predictions):
    prediction_path = get_paths()["prediction_path"]
    np.savetxt(prediction_path, predictions, fmt="%d")