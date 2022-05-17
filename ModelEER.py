import random
from os import listdir
from os.path import isdir

import pandas as pd
import sklearn
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from random import choice

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve, train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


if __name__ == '__main__':
    taps = pd.read_csv("features/ScaleCircleActivityIsUserTraining0.csv", delimiter=";", skipinitialspace=True)
    mm = make_pipeline(MinMaxScaler(), Normalizer())

    X_train, X_test, y_train, y_test = train_test_split(
        mm.fit_transform(taps.iloc[:, 2:]),
        LabelEncoder().fit_transform(taps['UT']), test_size=0.3, shuffle=True
    )

    model = RandomForestClassifier(max_depth=100, n_estimators=500)
    model.fit(X_train, y_train)
    yhat_test = model.predict(X_test)
    yhat_prob = model.predict_proba(X_test)
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, yhat_test)}\n"
    )

    a = []
    for i in range(len(taps.iloc[:, 2:])):
        random_user = mm.fit_transform(taps.iloc[:, 2:])[random.randint(0, len(taps.iloc[:, 2:])-1)]
        yhat_class = model.predict(random_user.reshape(1, -1))
        yhat_prob = model.predict_proba(random_user.reshape(1, -1))
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        intp = class_probability
        a.append(intp)

    # far is a list that we will save the False accaptance rate in each threshold
    # threshold is the list of thresold and it will go from 0% to 100%
    far = []
    threshold = []
    for i in range(100):
        num = 0

        for x in a:
            if x > i:
                num += 1
        # print(i,num)
        far.append(num)
        threshold.append(i)

    far = np.array(far)
    print('FAR: ', far)
    print('-----------------------------------------------------------')

    b = []
    for i in range(len(taps.iloc[:, 2:])):
        randomIndex = random.randint(0, i)
        random_face_pixels = mm.fit_transform(taps.iloc[:, 2:])[randomIndex]
        random_face_class = LabelEncoder().fit_transform(taps['UT'])[randomIndex]
        yhat_class = model.predict(random_face_pixels.reshape(1, -1))
        yhat_prob = model.predict_proba(random_face_pixels.reshape(1, -1))
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        if random_face_class == yhat_class:
            intp = class_probability
            # print(f'Predicted: {intp} %')
            b.append(intp)

    frr = []
    for i in range(100):
        num = 0

        for x in b:
            if x < i:
                num += 1
        # print(i,num)
        frr.append(num)

    frr = np.array(frr)
    print('FRR: ', frr)
    print('-----------------------------------------------------------')

    # FRR,FAR,EER
    fig, ax = plt.subplots()
    ax.plot(threshold, far, 'r--', label='FAR')
    ax.plot(threshold, frr, 'g--', label='FRR')
    plt.xlabel('Threshold')
    # plt.plot(i, EER, 'ro', label='EER')
    idx = np.argwhere(np.diff(np.sign(frr - far))).flatten()
    plt.plot(far[idx], frr[idx], 'ro')
    print("EER: ", far[idx], frr[idx])

    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    plt.show()
