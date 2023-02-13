import gin
import pandas as pd
import numpy as np
import tensorflow as tf

@gin.configurable
class dataset_creation:
    '''
    This class contains the basic functionality for creation of the dataset and also the windowing needed.
    The size of the window and its overlap is taken from the config file.
    '''
    def __init__(self, WINDOW_SIZE, SHIFT, DROP_REM, BATCH_SIZE):
        # Initialize the class's parameters
        self.WINDOW_SIZE = WINDOW_SIZE
        self.SHIFT = SHIFT
        self.DROP_REM = DROP_REM
        self.BATCH_SIZE = BATCH_SIZE

    def window_data_gen(self, input_dataframe):
        # Read in the input dataframe and split it into two dataframes - one for the input data and one for the labels
        input_dataframe = pd.read_csv(input_dataframe)
        df_x = input_dataframe.drop(input_dataframe.iloc[:, 6:], axis=1).astype(np.float64)
        df_y = input_dataframe.drop(input_dataframe.iloc[:, :6], axis=1).astype(np.int32)

        list_df = [df_x, df_y]

        for i in list_df:
            input_dataframe = i
            input_data = tf.data.Dataset.from_tensor_slices(input_dataframe)

            # create the sliding window dataset
            window_dataset = input_data.window(self.WINDOW_SIZE, shift=self.SHIFT, drop_remainder=self.DROP_REM)

            # map the window dataset to a dataset of window examples
            def create_window_examples(window):
                return window.batch(self.WINDOW_SIZE)

            if str(i) == str(df_x):
                window_examples_dataset_x = window_dataset.flat_map(create_window_examples)
            else:
                window_examples_dataset_y = window_dataset.flat_map(create_window_examples)

        # Return the window examples datasets for the input data and labels
        return window_examples_dataset_x, window_examples_dataset_y

    def dataset_gen(self, train_test_val_dir):
        # Generate the window examples datasets for the training, testing, and validation datasets
        self.train_df_x, self.train_df_y = self.window_data_gen(train_test_val_dir+'/train.csv')
        self.test_df_x, self.test_df_y = self.window_data_gen(train_test_val_dir+'/test.csv')
        self.val_df_x, self.val_df_y = self.window_data_gen(train_test_val_dir+'/val.csv')

    def batch_data_gen(self, train_test_val_dir):
        # Generate the datasets for training, testing, and validation
        self.dataset_gen(train_test_val_dir)
        # Zip the input data and labels together and batch them
        train_dataset = tf.data.Dataset.zip((self.train_df_x, self.train_df_y)).batch(self.BATCH_SIZE)
        test_dataset = tf.data.Dataset.zip((self.test_df_x, self.test_df_y)).batch(self.BATCH_SIZE)
        val_dataset = tf.data.Dataset.zip((self.val_df_x, self.val_df_y)).batch(self.BATCH_SIZE)
        # return the train, test and valdation dataset
        return train_dataset,test_dataset,val_dataset





