import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Input

class Layers:
    # Initialize the class with units and dropout rate as input
    def __init__(self, units, dropout_rate):
        self.units = units
        self.dropout_rate = dropout_rate

    # Create a method to build a BiLSTM model
    def bilstm(self):
        # Create BiLSTM model
        bilstm = tf.keras.Sequential([
            # First layer of BiLSTM
            Input(shape=(None, 6)),
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # Second layer of BiLSTM
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            Dense(13, activation='softmax')
        ])
        return bilstm

    # Create a method to build a GRU model
    def gru(self):
        # Create LSTM or GRU model
        gru = tf.keras.Sequential([
            Input(shape=(None, 6)),
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # Second layer of LSTM
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return gru

    # Create a method to build a LSTM model
    def lstm(self):
        # Create LSTM or GRU model
        lstm = tf.keras.Sequential([
            Input(shape=(None, 6)),
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # Second layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return lstm

    # Create a custom model
    def complex_model_1(self):
        complex_model_1 = tf.keras.Sequential([
            # First layer of LSTM
            Input(shape=(None, 6)),
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # second layer of GRU
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # third layer of BiLSTM
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_1

    # Create a custom model
    def complex_model_2(self):
        complex_model_2 = tf.keras.Sequential([
            # First layer of LSTM
            Input(shape=(None, 6)),
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # second layer of BiLSTM
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # third layer of GRU
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_2

    # Create a custom model
    def complex_model_3(self):
        complex_model_3 = tf.keras.Sequential([
            # First layer of GRU
            Input(shape=(None, 6)),
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # second layer of BiLSTM
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # third layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_3

    # Create a custom model
    def complex_model_4(self):
        complex_model_4 = tf.keras.Sequential([
            # First layer of GRU
            Input(shape=(None, 6)),
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # second layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # third layer of BiLSTM
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_4

    # Create a custom model
    def complex_model_5(self):
        complex_model_5 = tf.keras.Sequential([
            # First layer of BiLSTM
            Input(shape=(None, 6)),
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # second layer of GRU
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # third layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_5

    # Create a custom model
    def complex_model_6(self):
        complex_model_6 = tf.keras.Sequential([
            # First layer of BiLSTM
            Input(shape=(None, 6)),
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # second layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # third layer of GRU
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_6

    # Create a custom model
    def complex_model_1(self):
        complex_model_1 = tf.keras.Sequential([
            # First layer of BiLSTM
            Input(shape=(None, 6)),
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # second layer of GRU
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # third layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_1

    # Create a custom model
    def complex_model_1(self):
        complex_model_1 = tf.keras.Sequential([
            # First layer of BiLSTM
            Input(shape=(None, 6)),
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # second layer of GRU
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # third layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_1

    # Create a custom model
    def complex_model_1(self):
        complex_model_1 = tf.keras.Sequential([
            # First layer of BiLSTM
            Input(shape=(None, 6)),
            Bidirectional(LSTM(units=int(self.units), return_sequences=True)),
            Dropout(self.dropout_rate),
            # second layer of GRU
            GRU(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            # third layer of LSTM
            LSTM(units=int(self.units), return_sequences=True),
            Dropout(self.dropout_rate),
            Dense(units=13, activation='softmax')
        ])
        return complex_model_1


