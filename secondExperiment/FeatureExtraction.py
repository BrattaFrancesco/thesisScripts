import datetime
import time

import pandas as pd
import numpy as np
import os

def progress(percent, width=50):
    left = width * percent // 100
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"

    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)

def calculateSensor(userPath, time1, sensorName, k):
    sensorDf = pd.read_csv(userPath + "/" + sensorName + ".csv", delimiter=",",
                           skipinitialspace=True)

    while datetime.datetime.strptime(sensorDf.at[k, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f") < time1:
        k += 1
    return k, (np.sqrt(np.power(sensorDf.at[k, "X"], 2) + np.power(sensorDf.at[k, "Y"], 2) + np.power(sensorDf.at[k, "Z"], 2)))

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
        positionalSensors = [['ACCELEROMETER', 0], ['GYROSCOPE', 0], ['MAGNETOMETER', 0]]
        while i < len(df['ROW']):
            progress(int(i/len(df["ROW"])*100))
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

                        for positionalSensor in positionalSensors:
                            if positionalSensor[0] == 'ACCELEROMETER':
                                positionalSensor[1], value = calculateSensor(userPath, time1, positionalSensor[0], positionalSensor[1])
                                row.append(value)
                            elif positionalSensor[0] == 'GYROSCOPE':
                                positionalSensor[1], value = calculateSensor(userPath, time1, positionalSensor[0], positionalSensor[1])
                                row.append(value)
                            elif positionalSensor[0] == 'MAGNETOMETER':
                                positionalSensor[1], value = calculateSensor(userPath, time1, positionalSensor[0], positionalSensor[1])
                                row.append(value)
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
    second_experiment_dataset = '/home/francesco/PycharmProjects/Thesis/secondExperiment/Dataset/'
    os.chdir(second_experiment_dataset)
    tests = os.listdir(second_experiment_dataset)

    for test in tests:
        print("Test: ", test)
        processedPath = second_experiment_dataset + test + "/processed/"
        start_time = time.time()
        tapExtraction(processedPath)
        print("Completed in: " + str(time.time() - start_time) + "s")
