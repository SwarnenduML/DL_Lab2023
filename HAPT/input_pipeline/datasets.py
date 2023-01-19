import gin
import pandas as pd
import numpy as np
import tensorflow as tf

@gin.configurable
class dataset_creation:

    def __init__(self, window_size, shift, drop_rem, batch_size):
        self.window_size = window_size
        self.shift = shift
        self.drop_rem = drop_rem
        self.batch_size = batch_size

    def window_data_gen(self, input_dataframe):
        input_dataframe = pd.read_csv(input_dataframe)
        df_x = input_dataframe.drop(input_dataframe.iloc[:, 6:], axis=1).astype(np.float64)
        df_y = input_dataframe.drop(input_dataframe.iloc[:, :6], axis=1).astype(np.int32)

        list_df = [df_x, df_y]

        for i in list_df:
            input_dataframe = i
            input_data = tf.data.Dataset.from_tensor_slices(input_dataframe)

            # create the sliding window dataset
            window_dataset = input_data.window(self.window_size, shift=self.shift, drop_remainder=self.drop_rem)

            # map the window dataset to a dataset of window examples
            def create_window_examples(window):
                return window.batch(self.window_size)

            if str(i) == str(df_x):
                window_examples_dataset_x = window_dataset.flat_map(create_window_examples)
            else:
                window_examples_dataset_y = window_dataset.flat_map(create_window_examples)

        return window_examples_dataset_x, window_examples_dataset_y

    def dataset_gen(self, train_test_val_dir):
        self.train_df_x, self.train_df_y = self.window_data_gen(train_test_val_dir+'/train.csv')
        self.test_df_x, self.test_df_y = self.window_data_gen(train_test_val_dir+'/test.csv')
        self.val_df_x, self.val_df_y = self.window_data_gen(train_test_val_dir+'/val.csv')

    def batch_data_gen(self, train_test_val_dir):
        self.dataset_gen(train_test_val_dir)
        train_dataset = tf.data.Dataset.zip((self.train_df_x, self.train_df_y)).batch(self.batch_size)
        test_dataset = tf.data.Dataset.zip((self.test_df_x, self.test_df_y)).batch(self.batch_size)
        val_dataset = tf.data.Dataset.zip((self.val_df_x, self.val_df_y)).batch(self.batch_size)
        return train_dataset,test_dataset,val_dataset






