import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import Normalizer
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    featuresPath = "/home/francesco/PycharmProjects/Thesis/firstExperiment/features/"

    models = []
    models.append(RandomForestClassifier(max_depth=100, n_estimators=500))
    models.append(SVC(gamma=0.001, probability=True))
    models.append(KNeighborsClassifier(n_neighbors=3))
    models.append(DecisionTreeClassifier(max_depth=100))

    for model in models:
        print(model)
        for file in os.listdir(featuresPath):
            plt.figure(file)
            features = pd.read_csv(featuresPath + file, delimiter=";", skipinitialspace=True)
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            X = mm.fit_transform(features.iloc[:, 2:])
            y = LabelEncoder().fit_transform(features['UT'])
            lw = 1

            # Binarize the output
            y_bin = label_binarize(y, classes=[0, 1, 2, 3, 4, 5])
            n_classes = y_bin.shape[1]

            pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', model)])
            # or
            #clf = OneVsRestClassifier(LogisticRegression())
            #pipe= Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            y_score = cross_val_predict(pipe, X, y, cv=10, method='predict_proba')
            fpr = dict()
            tpr = dict()
            fnr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                fnr[i] = 1 - tpr[i]
                roc_auc[i] = auc(fpr[i], tpr[i])
            colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'black'])
            print(file.split(".")[0])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (AUC = {1:0.2f}) (EER = {2:0.2f}%)'
                         ''.format(i, roc_auc[i], fpr[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]*100))
                print("EER class ", i, ": ", fpr[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]*100, "%")
            print()
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic for multi-class data \n' + file.split(".")[0] + "\n" +
                      model.__str__())
            plt.legend(loc="lower right")
        plt.show()
