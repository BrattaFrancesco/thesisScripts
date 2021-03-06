import datetime
import time

import pandas as pd
import numpy as np
import os

def calculateSensor(intake, userPath, time1, sensorName):
    sensorDf = pd.read_csv(userPath + intake + "/" + sensorName + ".csv", delimiter=",",
                           skipinitialspace=True)
    k = 0
    while datetime.datetime.strptime(sensorDf.at[k, 'DATE\TIME'].split(" ")[1],
                                     "%H:%M:%S.%f") < time1:
        k += 1
    return (np.sqrt(np.power(sensorDf.at[k, "X"], 2) + np.power(sensorDf.at[k, "Y"], 2) + np.power(sensorDf.at[k, "Z"], 2)))

def tapExtraction(processedPath):
    # retrieve a list of all sensor directories
    users = os.listdir(processedPath)
    sensor = 'TapActivityIsUserTraining0.csv'
    rows = []

    for user in users:
        print("Working on: " + user)
        userPath = processedPath + user + '/'
        intakes = os.listdir(userPath)
        for intake in intakes:
            print("Intake: " + intake)
            df = pd.read_csv(userPath + intake + "/" + sensor, delimiter=",", skipinitialspace=True)

            # drop duplicated tap
            df['filter'] = np.where(df['SENSOR'].str.contains('ACTION_DOWN_TapActivityIsUserTraining0', na=False), True, False)
            actionDownDf = df[df['filter'] == True]
            actionDownDf.reset_index(inplace=True, drop=False)
            actionDownDf = actionDownDf.drop(columns=['index'])
            df = df.drop(columns=['filter'])
            i = 0
            dropList = []
            while i < len(actionDownDf['ROW']) - 1:
                if actionDownDf.at[i, 'N'] == actionDownDf.at[i + 1, 'N']:
                    dropList.append(actionDownDf.at[i, 'ROW'])
                i += 1
            print("Row number:" + str(i), "Rows to drop" + str(dropList))
            df = df.drop(df.index[dropList])
            df.reset_index(inplace=True, drop=False)

            # calculate features
            i = 0
            while i < len(df['ROW']):
                row = [user]
                j = i + 1
                if df.at[i, 'SENSOR'] == "ACTION_DOWN_TapActivityIsUserTraining0 ":
                    # tapPrecision
                    row.append(np.sqrt(np.power((df.at[i, 'x'] - df.at[i, 'xCenterButton']), 2) + np.power(
                        (df.at[i, 'y'] - df.at[i, 'yCenterButton']), 2)))
                    # pressure
                    row.append(df.at[i, 'p'])
                    # duration
                    while j < len(df['ROW']):
                        if df.at[j, 'SENSOR'] == 'ACTION_UP_TapActivityIsUserTraining0 ':
                            time1 = datetime.datetime.strptime(df.at[i, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")
                            time2 = datetime.datetime.strptime(df.at[j, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")
                            row.append((time2-time1).microseconds)

                            positionalSensors = ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']
                            for positionalSensor in positionalSensors:
                                if positionalSensor == 'ACCELEROMETER':
                                    row.append(calculateSensor(intake, userPath, time1, positionalSensor))
                                elif positionalSensor == 'GYROSCOPE':
                                    row.append(calculateSensor(intake, userPath, time1, positionalSensor))
                                elif positionalSensor == 'MAGNETOMETER':
                                    row.append(calculateSensor(intake, userPath, time1, positionalSensor))
                            break
                        j += 1
                    rows.append(row)
                i = j
            print("Length: " + str(len(rows)) + str(rows))
        print()
    featureDf = pd.DataFrame(rows, columns=["UT", "precision", "pressure", "duration", "acceleration", "rotation", "magneticField"])
    if not os.path.isdir(path_dataset + "/features/"):
        os.mkdir(path_dataset + "/features/")
    featureDf.to_csv(path_dataset + "/features/" + sensor, sep=';')

def swipePrecisionExtraction(processedPath):
    yAxis = 1113
    # retrieve a list of all sensor directories
    users = os.listdir(processedPath)
    sensor = 'SlideActivityIsUserTraining0.csv'
    features = []

    for user in users:
        print("Working on: " + user)
        userPath = processedPath + user + '/'
        intakes = os.listdir(userPath)
        for intake in intakes:
            print("Intake: " + intake)
            df = pd.read_csv(userPath + intake + "/" + sensor, delimiter=",", skipinitialspace=True)

            # let's create a dataframe with all the data that we need, based on the word in input
            df['filter'] = np.where(df['SENSOR'].str.contains('ACTION_DOWN_SlideActivityIsUserTraining0', na=False),
                                    True, False)
            actionDownDf = df[df['filter'] == True]
            actionDownDf.reset_index(inplace=True, drop=False)
            actionDownDf = actionDownDf.drop(columns=['index'])
            df = df.drop(columns=['filter'])

            i = 1
            realSwipeList = []
            while i < len(actionDownDf['ROW']) - 1:
                if actionDownDf.at[i, 'N'] != actionDownDf.at[i + 1, 'N']:
                    j = 1
                    while actionDownDf.at[i - j, 'x'] >= 400:
                        j += 1
                    realSwipeList.append(actionDownDf.at[i - j, 'ROW'])
                elif i == len(actionDownDf) - 2:
                    realSwipeList.append(actionDownDf.at[i, 'ROW'])
                    if actionDownDf.at[i, 'x'] >= 400:
                        print("There could be multiple next tap, check manually the data at line: ", i)
                i += 1
            print(i, realSwipeList)

            for row in realSwipeList:
                i = 0
                distances = []
                xSpeeds = []
                ySpeeds = []
                pressures = []
                accelerations = []
                rotations = []
                magneticFields = []
                time1 = datetime.datetime.strptime(df.at[row, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")
                time2 = datetime.datetime
                while not 'ACTION_UP_SlideActivityIsUserTraining0' in df[row:].at[row + i, 'SENSOR'] and i < len(df[row:]['SENSOR']) - 1:
                    xSpeeds.append(df.at[row + i, 'vX'])
                    ySpeeds.append(df.at[row + i, 'vY'])
                    pressures.append(df.at[row + i, 'p'])
                    distances.append(np.abs(yAxis - df.at[row + i, 'y']))
                    time2 = datetime.datetime.strptime(df.at[row + i, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")

                    positionalSensors = ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']
                    for positionalSensor in positionalSensors:
                        if positionalSensor == 'ACCELEROMETER':
                            accelerations.append(calculateSensor(intake, userPath, time2, positionalSensor))
                        elif positionalSensor == 'GYROSCOPE':
                            rotations.append(calculateSensor(intake, userPath, time2, positionalSensor))
                        elif positionalSensor == 'MAGNETOMETER':
                            magneticFields.append(calculateSensor(intake, userPath, time2, positionalSensor))
                    i += 1

                features.append([user, np.average(distances), np.average(xSpeeds[1:]), np.average(ySpeeds[1:]),
                                 np.average(pressures), np.average(xSpeeds[len(xSpeeds) - 5:]),
                                 np.average(ySpeeds[len(ySpeeds) - 5:]), (time2 - time1).microseconds,
                                 np.average(accelerations), np.average(rotations), np.average(magneticFields)])
        print()
    featureDf = pd.DataFrame(features, columns=["UT", "precision", "avgXSpeed", "avgYSpeed", "avgPressure",
                                                "xMedianSpeedOfLast5Points", "yMedianSpeedOfLast5Points", "duration",
                                                "avgAcceleration", "avgRotation", "avgMagneticField"])
    if not os.path.isdir(path_dataset + "/features/"):
        os.mkdir(path_dataset + "/features/")
    featureDf.to_csv(path_dataset + "/features/" + sensor, sep=';')


def scalingPrecisionExtraction(processedPath):
    xCenterPoint = 540
    yCenterPoint = 1098
    # retrieve a list of all sensor directories
    users = os.listdir(processedPath)
    sensor = 'ScaleCircleActivityIsUserTraining0.csv'
    finger0features = []
    finger1features = []

    for user in users:
        print("Working on: " + user)
        userPath = processedPath + user + '/'
        intakes = os.listdir(userPath)
        for intake in intakes:
            print("Intake: " + intake)
            df = pd.read_csv(userPath + intake + "/" + sensor, delimiter=",", skipinitialspace=True)

            # let's create a dataframe with all the data that we need, based on the word in input
            df['filter'] = np.where(
                df['SENSOR'].str.contains('ACTION_DOWN_ScaleCircleActivityIsUserTraining0', na=False), True, False)
            actionDownDf = df[df['filter'] == True]
            actionDownDf.reset_index(inplace=True, drop=False)
            actionDownDf = actionDownDf.drop(columns=['index'])
            df = df.drop(columns=['filter'])

            i = 1
            scaleList = []
            while i < len(actionDownDf['ROW']) - 1:
                if actionDownDf.at[i, 'N'] != actionDownDf.at[i + 1, 'N']:
                    scaleList.append(actionDownDf.at[i - 1, 'ROW'])
                elif i == len(actionDownDf) - 2:
                    scaleList.append(actionDownDf.at[i, 'ROW'])
                i += 1
            print(i, scaleList)

            finger0xSpeeds = []
            finger0ySpeeds = []
            finger0pressures = []
            finger0accelerations = []
            finger0rotations = []
            finger0magneticFields = []
            for row in scaleList:
                i = 0
                finger0time1 = datetime.datetime.strptime(df.at[row, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")
                finger0time2 = datetime.datetime
                while not 'ACTION_UP_ScaleCircleActivityIsUserTraining0' in df[row:].at[row + i, 'SENSOR'] and i < len(
                        df[row:]['SENSOR']) - 1:
                    if str(df.at[row + i, 'vX']) != 'nan':
                        finger0xSpeeds.append(df.at[row + i, 'vX'])
                    if str(df.at[row + i, 'vY']) != 'nan':
                        finger0ySpeeds.append(df.at[row + i, 'vY'])
                    if str(df.at[row + i, 'p']) != 'nan':
                        finger0pressures.append(df.at[row + i, 'p'])
                    finger0time2 = datetime.datetime.strptime(df.at[row + i, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")

                    positionalSensors = ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']
                    for positionalSensor in positionalSensors:
                        if positionalSensor == 'ACCELEROMETER':
                            finger0accelerations.append(calculateSensor(intake, userPath, finger0time2, positionalSensor))
                        elif positionalSensor == 'GYROSCOPE':
                            finger0rotations.append(calculateSensor(intake, userPath, finger0time2, positionalSensor))
                        elif positionalSensor == 'MAGNETOMETER':
                            finger0magneticFields.append(calculateSensor(intake, userPath, finger0time2, positionalSensor))
                    i += 1
                finger0features.append([user, np.sqrt(np.power((df.at[row + i, 'x'] - xCenterPoint), 2) +
                                np.power((df.at[row + i, 'y'] - yCenterPoint),2)),
                                np.average(finger0xSpeeds[1:]),
                                np.average(finger0ySpeeds[1:]),
                                np.average(finger0pressures), np.average(finger0xSpeeds[len(finger0xSpeeds) - 5:]),
                                np.average(finger0ySpeeds[len(finger0ySpeeds) - 5:]), (finger0time2 - finger0time1).microseconds,
                                np.average(finger0accelerations), np.average(finger0rotations), np.average(finger0magneticFields)])

            # let's create a dataframe with all the data that we need, based on the word in input
            df['filter'] = np.where(
                df['SENSOR'].str.contains('ACTION_POINTER_DOWN_ScaleCircleActivityIsUserTraining0', na=False), True,
                False)
            actionDownDf = df[df['filter'] == True]
            actionDownDf.reset_index(inplace=True, drop=False)
            actionDownDf = actionDownDf.drop(columns=['index'])
            df = df.drop(columns=['filter'])

            i = 0
            scaleList = []
            while i <= len(actionDownDf['ROW']) - 1:
                if i == len(actionDownDf) - 1:
                    scaleList.append(actionDownDf.at[i, 'ROW'])
                elif actionDownDf.at[i, 'N'] != actionDownDf.at[i + 1, 'N']:
                    scaleList.append(actionDownDf.at[i, 'ROW'])
                i += 1
            print(i, scaleList)

            finger1xSpeeds = []
            finger1ySpeeds = []
            finger1pressures = []
            finger1accelerations = []
            finger1rotations = []
            finger1magneticFields = []
            for row in scaleList:
                i = 0
                finger0time1 = datetime.datetime.strptime(df.at[row, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")
                finger0time2 = datetime.datetime
                while not 'ACTION_POINTER_UP_ScaleCircleActivityIsUserTraining0' in df[row:].at[row + i, 'SENSOR'] and i < len(
                        df[row:]['SENSOR']) - 1:
                    if str(df.at[row + i, 'vX']) != 'nan':
                        finger1xSpeeds.append(df.at[row + i, 'vX'])
                    if str(df.at[row + i, 'vY']) != 'nan':
                        finger1ySpeeds.append(df.at[row + i, 'vY'])
                    if str(df.at[row + i, 'p']) != 'nan':
                        finger1pressures.append(df.at[row + i, 'p'])
                    finger0time2 = datetime.datetime.strptime(df.at[row + i, 'DATE\TIME'].split(" ")[1], "%H:%M:%S.%f")

                    positionalSensors = ['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']
                    for positionalSensor in positionalSensors:
                        if positionalSensor == 'ACCELEROMETER':
                            finger1accelerations.append(calculateSensor(intake, userPath, finger0time2, positionalSensor))
                        elif positionalSensor == 'GYROSCOPE':
                            finger1rotations.append(calculateSensor(intake, userPath, finger0time2, positionalSensor))
                        elif positionalSensor == 'MAGNETOMETER':
                            finger1magneticFields.append(calculateSensor(intake, userPath, finger0time2, positionalSensor))
                    i += 1
                finger1features.append([user, np.sqrt(np.power((df.at[row + i, 'x'] - xCenterPoint), 2) +
                                np.power((df.at[row + i, 'y'] - yCenterPoint),2)),
                                np.average(finger1xSpeeds[1:]),
                                np.average(finger1ySpeeds[1:]),
                                np.average(finger1pressures), np.average(finger1xSpeeds[len(finger1xSpeeds) - 5:]),
                                np.average(finger1ySpeeds[len(finger1ySpeeds) - 5:]), (finger0time2 - finger0time1).microseconds,
                                np.average(finger1accelerations), np.average(finger1rotations), np.average(finger1magneticFields)])
        print()

    print("Creating the csv file...")
    i = 0
    features = []
    while i < len(finger0features):
        features.append(finger0features[i] + finger1features[i][1:])
        i += 1
    featureDf = pd.DataFrame(features, columns=["UT", "centerDistanceXf0", "avgXSpeedf0", "avgYSpeedf0", "avgPressuref0",
                                                "xMedianSpeedOfLast5Pointsf0", "yMedianSpeedOfLast5Pointsf0", "durationf0",
                                                "avgAccelerationf0", "avgRotationf0", "avgMagneticFieldf0",
                                                "centerDistanceXf1", "avgXSpeedf1", "avgYSpeedf1", "avgPressuref1",
                                                "xMedianSpeedOfLast5Pointsf1", "yMedianSpeedOfLast5Pointsf1",
                                                "durationf1", "avgAccelerationf1", "avgRotationf1", "avgMagneticFieldf1"])

    if not os.path.isdir(path_dataset + "/features/"):
        os.mkdir(path_dataset + "/features/")
    featureDf.to_csv(path_dataset + "/features/" + sensor, sep=';')


if __name__ == '__main__':
    path_dataset = '/home/francesco/PycharmProjects/Thesis/firstExperiment/'
    os.chdir(path_dataset)
    processedPath = path_dataset + "/processed/"

    exit = False
    while exit != True:
        choice = input('What kind of extraction you want to do?'
                       '\n\t1.Tap Precision'
                       '\n\t2.SwipePrecision'
                       '\n\t3.Scaling Precision'
                       '\n\t4.Do them all'
                       '\n\t-1.Exit'
                       '\nChoice: ')
        start_time = time.time()
        if choice == '1':
            tapExtraction(processedPath)
        elif choice == '2':
            swipePrecisionExtraction(processedPath)
        elif choice == '3':
            scalingPrecisionExtraction(processedPath)
        elif choice == '4':
            tapExtraction(processedPath)
            swipePrecisionExtraction(processedPath)
            scalingPrecisionExtraction(processedPath)
        elif choice == '-1':
            exit = True
            print("Bye")
        else:
            print("Wrong choice")
        print("Completed in: " + str(time.time() - start_time) + "s")
