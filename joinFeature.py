import datetime
import time

import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    featuresPath = "features/"
    tap = pd.read_csv(featuresPath + "TapActivityIsUserTraining0.csv", delimiter=";", skipinitialspace=True)
    slide = pd.read_csv(featuresPath + "SlideActivityIsUserTraining0.csv", delimiter=";", skipinitialspace=True)
    scale = pd.read_csv(featuresPath + "ScaleCircleActivityIsUserTraining0.csv", delimiter=";", skipinitialspace=True)

    columns = ["UT"]
    for column in tap.columns[2:]:
        columns.append(column + "Tap")

    for column in slide.columns[2:]:
        columns.append(column + "Slide")

    for column in scale.columns[2:]:
        columns.append(column + "Scale")

    final = pd.DataFrame(columns=columns)
    final.to_csv("/home/francesco/Desktop/x.csv", sep=';')
    print(final)
