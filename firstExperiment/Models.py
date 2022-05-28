# Standard scientific Python imports
import matplotlib.pyplot as plt
import pandas as pd

# Import datasets, classifiers and performance metrics
from numpy import mean, std
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

if __name__ == '__main__':
    task = list()
    task.append(["TapActivityIsUserTraining0", pd.read_csv("../features/TapActivityIsUserTraining0.csv", delimiter=";", skipinitialspace=True)])
    task.append(["SlideActivityIsUserTraining0", pd.read_csv("../features/SlideActivityIsUserTraining0.csv", delimiter=";", skipinitialspace=True)])
    task.append(["ScaleCircleActivityIsUserTraining0", pd.read_csv("../features/ScaleCircleActivityIsUserTraining0.csv", delimiter=";", skipinitialspace=True)])

    models = list()
    # models.append(LogisticRegression())
    # models.append(RidgeClassifier())
    # models.append(SGDClassifier())
    # models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier())
    models.append(ExtraTreeClassifier())
    # models.append(LinearSVC())
    models.append(SVC())
    models.append(GaussianNB())
    # models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier())
    models.append(ExtraTreesClassifier())
    # models.append(GaussianProcessClassifier())
    # models.append(GradientBoostingClassifier())
    # models.append(LinearDiscriminantAnalysis())
    # models.append(QuadraticDiscriminantAnalysis())

    for taps in task:
        print(taps[0])
        # prepare the cross-validation procedure
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        mm = make_pipeline(MinMaxScaler(), Normalizer())
        for model in models:
            # evaluate model
            scores = cross_val_score(model, mm.fit_transform(taps[1].iloc[:, 2:]),
                                     LabelEncoder().fit_transform(taps[1]['UT']), scoring='accuracy', cv=cv, n_jobs=-1)
            # report performance
            print('>%s:Accuracy: %.3f (%.3f)' % (model.__str__(), mean(scores), std(scores)))
        print()
    # # Split data into 50% train and 50% test subsets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     taps.iloc[:, 2:], taps['UT'], test_size=0.3, shuffle=True
    # )
    #
    # # Learn the digits on the train subset
    # clf.fit(X_train, y_train)
    #
    # # Predict the value of the digit on the test subset
    # predicted = clf.predict(X_test)
    #
    # print(
    #     f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # )
    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    #
    # plt.show()
