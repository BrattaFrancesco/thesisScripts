import datetime
import time

import pandas as pd
import numpy as np
import os

def progress(percent, n, width=40):
    left = width * percent // n
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{percent//100:.0f}%"

    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)

def calculateSensor(userPath, time1, sensorName):
    sensorDf = pd.read_csv(userPath + "/" + sensorName + ".csv", delimiter=",",
                           skipinitialspace=True)
    k = 0
    while datetime.datetime.strptime(sensorDf.at[k, 'DATE\TIME'].split(" ")[1],
                                     "%H:%M:%S.%f") < time1:
        k += 1
    return (np.sqrt(np.power(sensorDf.at[k, "X"], 2) + np.power(sensorDf.at[k, "Y"], 2) + np.power(sensorDf.at[k, "Z"], 2)))

def tapExtraction(processedPath):
    # retrieve a list of all sensor directories
    users = os.listdir(processedPath)
    rows = []

    for user in users:
        print("Working on: " + user)
        userPath = processedPath + user + '/'

        df = pd.read_csv(userPath + "/ACTION.csv", delimiter=",", skipinitialspace=True)

        # drop duplicated tap
        df['filter'] = np.where(df['SENSOR'].str.contains('ACTION_DOWN', na=False), True, False)
        actionDownDf = df[df['filter'] == True]
        actionDownDf.reset_index(inplace=True, drop=False)
        actionDownDf = actionDownDf.drop(columns=['index'])
        df = df.drop(columns=['filter'])

        # calculate features
        i = 0
        while i < len(df['ROW']):
            progress(i, len(df['ROW']))
            row = [user]
            j = i + 1
            if "ACTION_DOWN" in df.at[i, 'SENSOR']:
                # x and y
                row.append(df.at[i, 'x'])
                row.append(df.at[i, 'y'])

                if i != len(df["ROW"]):
                    #pressure
                    row.append(df.at[i+1, 'p'])
                    #surface
                    row.append(df.at[i+1, 's'])

                # duration
                while j < len(df['ROW']):
                    if 'ACTION_UP' in df.at[j, 'SENSOR']:
                        time1 = datetime.datetime.strptime(df.at[i, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")
                        time2 = datetime.datetime.strptime(df.at[j, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")
                        row.append((time2-time1).microseconds)

                        positionalSensors = ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']
                        for positionalSensor in positionalSensors:
                            if positionalSensor == 'ACCELEROMETER':
                                row.append(calculateSensor(userPath, time1, positionalSensor))
                            elif positionalSensor == 'GYROSCOPE':
                                row.append(calculateSensor(userPath, time1, positionalSensor))
                            elif positionalSensor == 'MAGNETOMETER':
                                row.append(calculateSensor(userPath, time1, positionalSensor))
                        break
                    j += 1
                rows.append(row)
            i = j
        print("Length: " + str(len(rows)) + str(rows))
    print()
    featureDf = pd.DataFrame(rows, columns=["UT", "x", "y", "pressure", "surface", "duration", "acceleration", "rotation", "magneticField"])
    if not os.path.isdir(processedPath + "../features/"):
        os.mkdir(processedPath + "../features/")
    featureDf.to_csv(processedPath + "../features/features.csv", sep=';')

if __name__ == '__main__':
    second_experiment = '/home/francesco/PycharmProjects/Thesis/secondExperimentDataset/'
    os.chdir(second_experiment)
    tests = os.listdir(second_experiment)

    for test in tests:
        print("Test: ", test)
        processedPath = second_experiment + test + "/processed/"
        start_time = time.time()
        tapExtraction(processedPath)
        print("Completed in: " + str(time.time() - start_time) + "s")
