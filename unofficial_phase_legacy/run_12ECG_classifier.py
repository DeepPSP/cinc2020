#!/usr/bin/env python

import numpy as np
import joblib
from keras.models import load_model
from get_12ECG_features import get_12ECG_features


def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    all_labels = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    valid_label_indices = [i for i,l in enumerate(all_labels) if l in classes]
    valid_labels = [l for i,l in enumerate(all_labels) if l in classes]

    class_map = {
        l: classes.index(l) for l in valid_labels
    }

    # Use your classifier here to obtain a label and score for each class.
    features = get_12ECG_features(data, header_data)
    pred_score = model.predict(features)[...,valid_label_indices]
    pred_score = np.mean(pred_score, axis=0)  # or np.max?

    threshold = 0.5
    pred_labels = np.where(pred_score>=threshold)[0]
    if len(pred_labels) == 0:
        pred_labels = np.array([np.argmax(pred_score)], dtype=int)

    for l in pred_labels:
        ln = valid_labels[l]
        current_label[class_map[ln]] = 1

    for i in range(num_classes):
        current_score[class_map[valid_labels[i]]] = pred_score[i]

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='weights-0.22loss.hdf5'
    loaded_model = load_model(filename)

    return loaded_model


def run_12ECG_classifier_old(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    label = model.predict(feats_reshape)
    score = model.predict_proba(feats_reshape)

    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score
