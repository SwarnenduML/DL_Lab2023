# Team 14
- Swarnendu Sengupta (st180170)
- Lakshay Choudhary (st176718)

# How to run the code
The main.py can be run without any FLAG. The config file needs to be set and given correct parameters.
All folders must be downloaded.

Simple `python main.py` would work.

The parameters have to be set in the config.gin file present in the config folder.
The heading under "Main" describes the model that is needed to be built. It also tells if optimization is to 
be used or not. In this scenario, grid search was done. 
It also specifies the model on which the run is to be done. Grad CAM is the visualization technique that is to be used. It is also set in a True/False condition.

The "DatasetLoader" has the parameters referring to the batch size, clahe preprocessing output storage directory, input directory, use of augmentation, clahe equalization and upsampling.
The "Train and Test" gives the parameter has the parameters of number of epochs to run along with early stopping criteria and also a chance to save the models in a folder.

The "Tuning" determines the number of layers to be frozen for each model that has been considered.
In the "Visualization", number of images, size of images, the source of images and the position for output storage can be placed.
In the "Architecture", the units in case of a baseline can be specified along with the dropout rates for the other transfer learning models.
In the "Hyperparameter Optimization", the units of baseline are cosidered to be 16 and 32, dropout rate to be 0.1 and 0.3 and the learning rate to be 1e-1 and 1e-3.


# Results
In this case, due to unbalanced dataset F1-score and accuracy was considered. Clahe and pretrained EfficientNet provided the best results out of all the other available options.