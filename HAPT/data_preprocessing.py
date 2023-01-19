import gin
import os
import pandas as pd
import numpy as np

@gin.configurable
class DatasetLoader:
    '''A class to load, sample, parse and save datasets.'''

    def __init__(self, input_dir, output_dir, train_test_val_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        col_names = ['exp', 'user', 'activity', 'start', 'end']
        self.labels = pd.read_csv(input_dir+'labels.txt', sep = ' ',names = col_names, header = None)
        self.train_test_val_dir = train_test_val_dir

    def data_with_activity_gen(self, concat_df, filename, counter):
        concat_df.insert(6, "activity", value=0.0)
        start_values = self.labels.loc[self.labels.exp == counter].start.values
        end_values = self.labels.loc[self.labels.exp == counter].end.values
        activity_values = self.labels.loc[self.labels.exp == counter].activity.values

        con = np.concatenate((start_values, end_values))
        con.sort()

        j = 0
        for i in range(0, len(con), 2):
            concat_df.loc[(concat_df.index >= con[i]) & (concat_df.index <= con[i + 1]), 'activity'] = \
            activity_values[j]
            j += 1

        concat_df.to_csv(os.path.join(self.output_dir, filename[-16:]), index=False)

    def concat_data_gen(self):
        print("Data Generated")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        files_list = os.listdir(self.input_dir)
        files_list.sort()

        len_files = int((len(files_list) - 1) / 2)

        counter = 0
        for files in files_list[:len_files]:
            filename = files

            for files in files_list[len_files + counter:-1]:
                filename_df = pd.read_csv(self.input_dir + filename, sep=' ', names=['acc_x', 'acc_y', 'acc_z'],
                                          header=None)
                files_df = pd.read_csv(self.input_dir + files, sep=' ', names=['gyro_x', 'gyro_y', 'gyro_z'],
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

    def train_test_val_gen(self, files, filename):

        if not os.path.exists(self.train_test_val_dir):
            os.makedirs(self.train_test_val_dir)

        df = pd.DataFrame()
        for i in files:
            df_temp = pd.read_csv(self.output_dir + i)
            df = pd.concat([df, df_temp])

        df.to_csv(os.path.join(self.train_test_val_dir, filename), index=False)

    def load(self):
        self.concat_data_gen()
        # Train dataset: user-01 to user-21
        # Validation dataset: user-28 to user-30
        # Test dataset: user-22 to user-27

        train_files = []
        val_files = []
        test_files = []

        files = os.listdir(self.output_dir)
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
        print("Data generated in "+str(self.train_test_val_dir))
        return self.train_test_val_dir





