import gin
from models.layers import Layers

@gin.configurable
class Model:
    def __init__(self, units, dropout_rate):
        self.units = units
        self.dropout_rate = dropout_rate

    def base_model_bilstm(self):
        # Create BiLSTM model
        models = Layers(self.units, self.dropout_rate)
        model_bilstm = models.bilstm()

        return model_bilstm

    def base_model_gru(self):
        models = Layers(self.units, self.dropout_rate)
        model_gru = models.gru()
        return model_gru

    def base_model_lstm(self):
        models = Layers(self.units, self.dropout_rate)
        model_lstm = models.lstm()
        return model_lstm