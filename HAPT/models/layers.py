import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Input

class Layers:
    def __init__(self, units, dropout_rate):
        self.units = units
        self.dropout_rate = dropout_rate

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