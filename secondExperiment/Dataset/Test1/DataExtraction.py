import time

import pandas as pd
import numpy as np
import os
import re
import glob
sensor_keywords = ['ACTION', 'ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']  # change this

# This function determs the columns number of the database
def findNumberOfColumns(a):
    max = 0
    for row in a:
        if len(row.split(' ')) > max:
            max = len(row.split(' '))
    return max


if __name__ == '__main__':
    path_dataset = '/home/francesco/PycharmProjects/Thesis/secondExperiment/Dataset/Test1'
    os.chdir(path_dataset)
    sourcePath = path_dataset + "/raw/"
    # retrieve a list of all raw sensor files
    files = os.listdir(sourcePath)

    start_time = time.time()
    #Split raw sensor data into different files
    for file in files:
        print("Working on " + file)
        for word in sensor_keywords:
            if file != 'survey':
                df = pd.read_csv(sourcePath + "/" + file, delimiter="-", skipinitialspace=True, names=["DATE\TIME", "SENSOR", "VALUE"])

                # let's create a dataframe with all the data that we need, based on the word in input
                df['mask_df1'] = np.where(df['SENSOR'].str.contains(word, na=False), True, False)
                acc_activities = df[df['mask_df1'] == True]
                acc_activities.reset_index(inplace=True, drop=False)

                # let's create a better formed dataframe that contains just the value of the sensor mentioned before
                df1 = pd.DataFrame(columns=list(range(0, findNumberOfColumns(acc_activities['VALUE']) + 2)))

                i = 0
                while i < len(acc_activities['SENSOR']):
                    values = re.split(' |,', acc_activities['VALUE'][i])
                    # let's write the timestamp and the type of sensor
                    df1.loc[df.index[i], 0] = acc_activities['DATE\TIME'][i]
                    df1.loc[df.index[i], 1] = acc_activities['SENSOR'][i]

                    # now we can write the actual input values
                    j = 2
                    while j < len(values) + 2:
                        df1.loc[df.index[i], j] = values[j - 2]
                        j += 1
                    i += 1

                w1, w2 = str(file).split(".")
                if not os.path.isdir(path_dataset + '/processed/' + w1 + '/'):
                    if not os.path.isdir(path_dataset + '/processed/'):
                        os.mkdir(path_dataset + '/processed/')
                    os.mkdir(path_dataset + '/processed/' + w1 + '/')

                df1.to_csv(path_dataset + '/processed/' + w1 + '/' + word + '.csv')
    print()

    #Edit the tables adding headers and removing useless label
    processedPath = path_dataset + "/processed/"
    users = os.listdir(processedPath)

    for user in users:
        print("Edit headers...")
        print("Working on " + user)
        userPath = processedPath + user + '/'
        sensors = os.listdir(userPath)
        for sensor in sensors:
            df = pd.read_csv(userPath + "/" + sensor, delimiter=",", skipinitialspace=True)
            df = df.iloc[:, 0:-3]
            if 'ACCELEROMETER' in sensor or 'GYROSCOPE' in sensor or 'MAGNETOMETER' in sensor:
                df.columns = ['ROW', 'DATE\TIME', 'SENSOR', 'X', 'Y', 'Z']
            else:
                columns = ['ROW', 'DATE\TIME', 'SENSOR']
                i = 0
                while df.iloc[i].array.isin([np.nan]).any():
                    i += 1

                for field in df.iloc[i][3:]:
                    name, value = str(field).split(":")
                    columns.append(name)
                df.columns = columns

                for row in df['ROW']:
                    if not 'MOVE' in df.loc[row][2]:
                        df.loc[row] = df.iloc[row, 0:-4]
                    for field in df.loc[row][3:]:
                        if str(field) != 'nan':
                            column, value = str(field).split(":")
                            df.at[row, column] = value

                for column in df.columns[3:]:
                    i = 0
                    for row in df[column]:
                        if len(str(row).split(":")) > 1:
                            df.at[i, column] = ''
                        i += 1
            df.to_csv(userPath + "/" + sensor, index=False)
print("Completed in: " + str(time.time() - start_time) + "s")
