import gin
import os
import pandas as pd
import numpy as np

@gin.configurable
class DatasetLoader:
    '''create a class to handle loading, sampling, parsing, and saving datasets.'''

    def __init__(self, INPUT_DIR, OUTPUT_DIR, TRAIN_TEST_VAL_DIR):
        # initialize the class with input and output directories, and the directory for train, test and validation sets
        self.INPUT_DIR = INPUT_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
        col_names = ['exp', 'user', 'activity', 'start', 'end']
        self.labels = pd.read_csv(INPUT_DIR+'labels.txt', sep = ' ',names = col_names, header = None)
        self.TRAIN_TEST_VAL_DIR = TRAIN_TEST_VAL_DIR

    # function to add activity column to the dataset and save it to the output directory
    def data_with_activity_gen(self, concat_df, filename, counter):
        # add a new column for activity
        concat_df.insert(6, "activity", value=0.0)
        start_values = self.labels.loc[self.labels.exp == counter].start.values
        end_values = self.labels.loc[self.labels.exp == counter].end.values
        activity_values = self.labels.loc[self.labels.exp == counter].activity.values

        con = np.concatenate((start_values, end_values))
        con.sort()

        j = 0
        # assign activity values to corresponding time intervals
        for i in range(0, len(con), 2):
            concat_df.loc[(concat_df.index >= con[i]) & (concat_df.index <= con[i + 1]), 'activity'] = \
            activity_values[j]
            j += 1

        # save the activity
        concat_df.to_csv(os.path.join(self.OUTPUT_DIR, filename[-16:]), index=False)

    # function to concatenate data and normalize it
    def concat_data_gen(self):
        print("Data Generated")
        # create output directory if it does not exist
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

        # get a list of all files in the input directory
        files_list = os.listdir(self.INPUT_DIR)
        files_list.sort()

        len_files = int((len(files_list) - 1) / 2)

        counter = 0
        # loop through files to concatenate and process data
        for files in files_list[:len_files]:
            filename = files
            # concatenate the accelerometer and the gyroscope data
            for files in files_list[len_files + counter:-1]:
                filename_df = pd.read_csv(self.INPUT_DIR + filename, sep=' ', names=['acc_x', 'acc_y', 'acc_z'],
                                          header=None)
                files_df = pd.read_csv(self.INPUT_DIR + files, sep=' ', names=['gyro_x', 'gyro_y', 'gyro_z'],
                                       header=None)
                concat_df = pd.concat([filename_df, files_df], axis=1)
                col_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                for column in col_names:
                    concat_df[column] = (concat_df[column] - concat_df[column].mean()) / concat_df[column].std()
                counter += 1
                self.data_with_activity_gen(concat_df, filename, counter)
                break

    def namestr(self, obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    # generate the data from files as per need
    def train_test_val_gen(self, files, filename):
        if not os.path.exists(self.TRAIN_TEST_VAL_DIR):
            os.makedirs(self.TRAIN_TEST_VAL_DIR)

        df = pd.DataFrame()
        for i in files:
            df_temp = pd.read_csv(self.OUTPUT_DIR + i)
            df = pd.concat([df, df_temp])

        df.to_csv(os.path.join(self.TRAIN_TEST_VAL_DIR, filename), index=False)

    # load the dataset and then split it into train, test, valid dataset
    def load(self):
        self.concat_data_gen()
        # Train dataset: user-01 to user-21
        # Validation dataset: user-28 to user-30
        # Test dataset: user-22 to user-27

        train_files = []
        val_files = []
        test_files = []

        files = os.listdir(self.OUTPUT_DIR)
        files.sort()

        for i in files:
            if int(i[-6:-4]) <= 21:
                train_files.append(i)
            elif int(i[-6:-4]) >= 28 and int(i[-6:-4]) <= 30:
                val_files.append(i)
            else:
                test_files.append(i)
        print("train, test, val split completed")
        self.train_test_val_gen(val_files, 'val.csv')
        self.train_test_val_gen(train_files, 'train.csv')
        self.train_test_val_gen(test_files,'test.csv')
        print("Data generated in "+str(self.TRAIN_TEST_VAL_DIR))
        return self.TRAIN_TEST_VAL_DIR





