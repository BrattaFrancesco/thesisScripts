import pandas as pd
import numpy as np
import os
import glob


def tapPrecisionExtraction(processedPath):
    # retrieve a list of all sensor directories
    users = os.listdir(processedPath)
    sensor = 'ACTION_DOWN_TapActivityIsUserTraining0.csv'

    for user in users:
        print("Working on: " + user)
        userPath = processedPath + user + '/'
        intakes = os.listdir(userPath)
        for intake in intakes:
            print("Intake: " + intake)
            df = pd.read_csv(userPath + intake + "/" + sensor, delimiter=",", skipinitialspace=True)
            i = 0
            dropList = []
            while i < len(df['ROW']) - 1:
                if df.at[i, 'N'] == df.at[i + 1, 'N']:
                    dropList.append(df.index[i])
                i += 1
            print(i, dropList)
            df = df.drop(df.index[dropList])
            df.reset_index(inplace=True, drop=False)

            i = 0
            precisionList = []
            while i < len(df['ROW']):
                precisionList.append(np.sqrt(np.power((df.at[i, 'x'] - df.at[i, 'xCenterButton']), 2) + np.power(
                    (df.at[i, 'y'] - df.at[i, 'yCenterButton']), 2)))
                i += 1
            print(precisionList)
        print()


def swipePrecisionExtraction(processedPath):
    yAxis = 1113
    # retrieve a list of all sensor directories
    users = os.listdir(processedPath)
    sensor = 'SlideActivityIsUserTraining0.csv'

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

            # print(actionDownDf)
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

            avgDistances = []
            for row in realSwipeList:
                i = 0
                distances = []
                while not 'ACTION_UP_SlideActivityIsUserTraining0' in df[row:].at[row + i, 'SENSOR'] and i < len(
                        df[row:]['SENSOR']) - 1:
                    distances.append(np.abs(yAxis - df.at[row + i, 'y']))
                    i += 1
                avgDistances.append(np.average(distances))
            print(avgDistances)
        print()


def scalingPrecisionExtraction(processedPath):
    xCenterPoint = 540
    yCenterPoint = 1098
    # retrieve a list of all sensor directories
    users = os.listdir(processedPath)
    sensor = 'ScaleCircleActivityIsUserTraining0.csv'

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

            finger0Distances = []
            for row in scaleList:
                i = 0
                while not 'ACTION_UP_ScaleCircleActivityIsUserTraining0' in df[row:].at[row + i, 'SENSOR'] and i < len(
                        df[row:]['SENSOR']) - 1:
                    i += 1
                finger0Distances.append(np.sqrt(
                    np.power((df.at[row + i, 'x'] - xCenterPoint), 2) + np.power((df.at[row + i, 'y'] - yCenterPoint),
                                                                                 2)))
            print(finger0Distances)

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

            finger1Distances = []
            for row in scaleList:
                i = 0
                while not 'ACTION_POINTER_UP_ScaleCircleActivityIsUserTraining0' in df[row:].at[
                    row + i, 'SENSOR'] and i < len(df[row:]['SENSOR']) - 1:
                    i += 1
                finger1Distances.append(np.sqrt(
                    np.power((df.at[row + i, 'x'] - xCenterPoint), 2) + np.power((df.at[row + i, 'y'] - yCenterPoint), 2)))
            print(finger1Distances)
        print()


if __name__ == '__main__':
    path_dataset = '/home/francesco/PycharmProjects/Thesis'
    os.chdir(path_dataset)
    processedPath = path_dataset + "/processed/"

    exit = False
    while exit != True:
        choice = input('What kind of extraction you want to do?'
                       '\n\t1.Tap Precision.'
                       '\n\t2.SwipePrecision'
                       '\n\t3.Scaling Precision'
                       '\n\t-1.Exit'
                       '\nChoice: ')
        if choice == '1':
            tapPrecisionExtraction(processedPath)
        elif choice == '2':
            swipePrecisionExtraction(processedPath)
        elif choice == '3':
            scalingPrecisionExtraction(processedPath)
        elif choice == '-1':
            exit = True
            print("Bye")
        else:
            print("Wrong choice")