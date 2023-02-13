# Team 14
- Swarnendu Sengupta (st180170)
- Lakshay Choudhary (st176718)

# How to run the code
The main.py can be run without any FLAGs. The parameters in the config.gin file has to be set in order to run the script.
All the folders need to be downloaded for the script to work.

Simple `python main.py` would work.

The parameters have to be set in the config.gin file present in the config folder.
The heading under "Main" describes the model that is needed to be built. It also tells if Bayesian optimization is to 
be used or not. It also specifies the start time and the end time on which visualization is to be done.

The "Dataset Creation" has the parameters referring to the shifting, windowing and also the batch size.
The " Bayesian Optimization" gives the parameter between which the Bayesian optimization is to be performed along with 
the number of epochs, early stopping and the number of points to be considered for Bayesian optimisation.

The "Architectures" determines the number of units and the dropout rate if the Bayesian optimisation is set to False in 
the "Main".
In the Training, epochs, early stopping criteria and also the basic learning rate can be set.
In the Input pipeline, the various directories are specified in which all the respective files would be created.
In the Visualization, the output directory where the visualization is to be stored is given.


# Results
Several models were tried out (LSTM, GRU, BiLSTM and their combinations at various levels). In this case, accuracy was the metric that was calculated and taken into account.
A combination model with LSTM, BiLSTM and GRU performed marginally better than the others. The number of units per layer, learning rate and dropout after each layer was tuned by Bayesian optimisation.
