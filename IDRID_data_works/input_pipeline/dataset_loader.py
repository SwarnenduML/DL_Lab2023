import gin
import os
import pathlib
import logging
import tensorflow as tf
#import tensorflow_datasets as tfds
import pandas as pd
import cv2
import numpy as np
#import typing
import tensorflow_addons as tfa
from tensorflow.image import ResizeMethod


#from input_pipeline.preprocessing import preprocess_tensorflow_dataset, augment_tensorflow_dataset, \
#                                         preprocess_augment, preprocess_resize, \
#                                         split_training_dataset, split_training_dataset_for_sampling, \
#                                         oversample, undersample, get_dataset_size, join_dataset

# Flag to display the dataset distibution after dataset creation.
PRINT_DATASET_DISTRIBUTIONS = False


@gin.configurable
class DatasetLoader:
    '''A class to load, sample, parse and save datasets.'''

    def __init__(self, dataset_name, dataset_directory, selected_model,  # <- configs.
                 training_dataset_ratio, sample_equal, augment, batch_size, equalization,            # <- configs.
                 output_dataset_directory):
        '''Parameters
        ----------
        dataset_name : str
            Name of the dataset. Supported datasets: 'idrid', 'eyepacs' and 'mnist'.
        dataset_directory : str
            Path to where the dataset data is stored.
        tfrecords_directory : str
            Destination for tfrecords files.
        training_dataset_ratio : float
            Ratio discribing how to split the training dataset into the traning and validation datasets.
        '''
        self.selected_model = selected_model
        if self.selected_model.lower() not in ['baseline', 'inception_v3', 'resnet_50', 'efficient', 'none']:
            raise ValueError(f"{self.selected_model} not available")

        if batch_size>0 and isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            raise ValueError(f"Received batch size {batch_size}. Accepted size > 0. Preferrably in powers of 2")

        self.dataset_name = dataset_name
        accepted_dataset_names = ('idrid', 'eyepacs', 'mnist')
        if self.dataset_name not in accepted_dataset_names:
            raise ValueError(
                f"Received invalid dataset name: '{self.dataset_name}', accepted dataset names: {accepted_dataset_names}")

        self.dataset_directory = pathlib.Path(dataset_directory)
        if (not self.dataset_directory.exists() ) or (not self.dataset_directory.is_dir() ):
            raise ValueError(f"Received invalid dataset directory: '{self.dataset_directory}'.")
        
        self.output_dataset_directory = pathlib.Path(output_dataset_directory)
        if (not self.output_dataset_directory.exists() ) or (not self.output_dataset_directory.is_dir() ):
            raise ValueError(f"Received invalid output dataset directory: '{self.output_dataset_directory}'.")
        
        if isinstance(training_dataset_ratio, float) and (0.0 < training_dataset_ratio <= 1.0):
            self.training_dataset_ratio = training_dataset_ratio
        else:
            raise ValueError(
                f'The training dataset split ratio has to be: 0.0 < ratio <= 1.0. Received ratio: {training_dataset_ratio}')

        
        if isinstance(sample_equal,bool):
            self.sample_equal = sample_equal
        else:
            raise ValueError(f"(Required True or False. Received {sample_equal})")

        if isinstance(augment,bool):
            self.augment_needed = augment
        else:
            raise ValueError(f"(Required True or False. Received {augment})")

        if isinstance(equalization,bool):
            self.equalization = equalization
        else:
            raise ValueError(f"(Required True or False. Received {equalization})")
        
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        

    def parse_function(self, filename, label):
        image_string = tf.io.read_file( str(self.dataset_directory) + filename)

        #Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.io.decode_jpeg(image_string, channels=3)

        if self.selected_model.lower() == 'baseline' or self.selected_model.lower() == 'efficient':
            # This will convert to float values in [0, 1]
            image = tf.image.convert_image_dtype(image, tf.float32)
        else:
            image = tf.cast(image, tf.float32)
            if self.selected_model.lower() == 'inception_v3':
                # This will convert to float values in [-1,1]
                image = tf.keras.applications.inception_v3.preprocess_input(image)
            elif self.selected_model.lower() =='resnet_50':
                # This will convert to float values in [-1,1]
                image = tf.keras.applications.resnet_v2.preprocess_input(image)

        # CCreating a bounding box to get more focus on the image
        image = tf.image.crop_to_bounding_box(image, 0, 200, 2848, 3550)

        image = tf.image.resize_with_pad(image,256, 256, method=ResizeMethod.BILINEAR, antialias=False)
        return image, label

    def make_ds(self,images_paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((images_paths, labels))
        ds = ds.shuffle(5000)
        ds = ds.map(self.parse_function, num_parallel_calls= self.AUTOTUNE)
        return ds


    def create_datasets(self):
        '''Create datasets.
        - For IDRID: Parse the data and create tfrecord files.
        - For EYEPACS and MNIST: Do nothing, the files are just loaded in `self.load_dataset()`.

        Returns
        -------
        Tuple of string
            List of directories in `self.tfrecords_directory`, these are the names of created datasets
            which can be loaded with `self.load_dataset()`.
        '''
        
        if self.dataset_name == 'idrid':
            logging.info(f"Preparing dataset '{self.dataset_name}'...")
            if self.equalization:
                self._save_image()
            train_ds, test_ds, valid_ds = self._create_idrid_dataset()
        elif self.dataset_name == 'eyepacs':
            logging.info("Sampling is not implemented for the 'eyepacs' dataset.")
        elif self.dataset_name == 'mnist':
            logging.info("Sampling is not implemented for the 'MNIST' dataset.")

        return train_ds, test_ds, valid_ds
    
    def clahe_processed_dataset_gen(self, input_dir, output_dir):
        # Create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through the images in the input directory
        for filename in os.listdir(input_dir):
            if filename.startswith("IDR"):
                # Read the image
                img = cv2.imread(os.path.join(input_dir, filename))
                img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

                # Apply CLAHE to the image
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                img_lab[...,0] = clahe.apply(img_lab[...,0])
                img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)

                # Save the processed image
                cv2.imwrite(os.path.join(output_dir, filename), img_clahe)

    def _save_image(self):
        input_dir = str(self.dataset_directory)
        output_dir = str(self.output_dataset_directory)
        input_train_path = input_dir+"/images/train"
        input_test_path = input_dir+"/images/test"
        output_train_path = output_dir+"/images/train"
        output_test_path = output_dir+"/images/test"
        if len(os.listdir(output_train_path))<413 or (not os.path.exists(output_train_path)) :
            self.clahe_processed_dataset_gen(input_train_path, output_train_path)
        logging.info(f"training clahe done")
        if len(os.listdir(output_test_path))<103 or (not os.path.exists(output_test_path)):
            self.clahe_processed_dataset_gen(input_test_path, output_test_path)
        logging.info(f"testing clahe done")

    def augment(self, images_labels, seed):
        images, labels= images_labels
        # Make a new seed.
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        images = tf.image.stateless_random_flip_left_right(images, seed =new_seed)
        k = np.random.uniform(low=1.0, high=20.0)
        images = tfa.image.rotate(images, tf.constant(np.pi / k))
        images = tf.image.stateless_random_flip_up_down(images, seed=new_seed)
        k = np.random.uniform(low=1.0, high=20.0)
        images = tfa.image.rotate(images, 20)
        # images = tf.image.stateless_random_contrast(images , 0.2, 0.5, seed = new_seed)
#         images = tf.image.stateless_random_brightness(images, 0.2, seed = new_seed)
        # images = tf.image.stateless_random_saturation(images, lower=0.1, upper=0.9, seed= new_seed)
#         images = tf.image.stateless_random_jpeg_quality(images, 60, 90, seed = new_seed)
        return images, labels
    
    def _create_idrid_dataset(self) :
        '''Read image and label files from data directory.

        Returns
        -------
        tuple of two tf.data.Dataset
            The training and test dataset, the training set is not yet split.
        '''
        logging.info(f"Preparing file loading from {self.dataset_directory}...")

        # Parse the CSV files
        labels_base_path = str(self.dataset_directory) + '/labels'
        train_label_data_dir = labels_base_path + "/train.csv"
        test_label_data_dir = labels_base_path + "/test.csv"
        #logging.info(test_label_data_dir)

        column_names = ["Image name", "Retinopathy grade", "Risk of macular edema"]
        training_data = pd.read_csv(train_label_data_dir, names=column_names, usecols=column_names)
        training_data = training_data.iloc[1:]  # Removing the first row as it contains header
        training_data['Retinopathy grade'].replace({'0':0,'1':0,'2':1,'3':1,'4':1}, inplace = True)# Changing from multiclass to binary class

        test_data = pd.read_csv(test_label_data_dir, names=column_names, usecols=column_names)
        test_data = test_data.iloc[1:]  # Removing the first row as it contains header
        test_data['Retinopathy grade'].replace({'0':0,'1':0,'2':1,'3':1,'4':1}, inplace = True) # Changing from multiclass to binary class
        #logging.info(test_data.head())
        #logging.info(f"Conversion completed")
        total_data_amount = len(training_data)
        
        train_data = training_data.loc[:int(self.training_dataset_ratio * total_data_amount)]
        val_data = training_data.loc[int(self.training_dataset_ratio * total_data_amount) + 1 : ]
        #logging.info(len(train_data))
        #logging.info(len(val_data))
        #logging.info(total_data_amount)
        #logging.info(val_data.head())

        # Creating pos and neg train dataset for later balancing
        pos_train_data = train_data[train_data['Retinopathy grade'] == 1]
        neg_train_data = train_data[train_data['Retinopathy grade'] == 0]
        #logging.info(f"Positive and negative datasets created")
        #logging.info(pos_train_data.head())
        
        pos_images_paths = '/images/train/'+pos_train_data['Image name'].values + '.jpg'
        neg_images_paths = '/images/train/'+neg_train_data['Image name'].values + '.jpg'
        val_images_path = '/images/train/'+ val_data['Image name'].values + '.jpg'
        #logging.info(f"paths found")
#        logging.info(pos_images_paths.head())

        pos_labels = pos_train_data['Retinopathy grade'].values
        neg_labels = neg_train_data['Retinopathy grade'].values
        val_labels = val_data['Retinopathy grade'].values
        #logging.info(f"Values known")
        #logging.info(pos_labels)

        if self.equalization:
            self.dataset_directory = self.output_dataset_directory
        pos_ds = self.make_ds(pos_images_paths, pos_labels)
        neg_ds = self.make_ds(neg_images_paths, neg_labels)
        val_ds = self.make_ds(val_images_path, val_labels).batch(self.batch_size)
        #logging.info(f"make_ds works")
#        for i,j in pos_ds.take(1):
#            logging.info(j)
        
        test_images_path = '/images/test/'+test_data['Image name'].values + '.jpg'
        test_labels = test_data['Retinopathy grade'].values
        test_ds = self.make_ds(test_images_path, test_labels).batch(len(test_labels))
        #logging.info(f"All basic datasets created")

        # Oversampling neg_ds to pos_ds
        sample_neg_ds = neg_ds.take(len(pos_ds) - len(neg_ds))
        neg_ds = neg_ds.concatenate(sample_neg_ds)


        if self.sample_equal:
            train_dataset = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
            train_ds = train_dataset.shuffle(600)#----------------------------------------------------------------------
            if self.augment:
                counter = tf.data.experimental.Counter()#tf.data.Dataset.counter()
                train_ds = tf.data.Dataset.zip((train_dataset, (counter, counter)))
                train_ds = train_ds.map(self.augment, num_parallel_calls = self.AUTOTUNE)
                logging.info(f"Datasets corresponding to {self.sample_equal} sampling and augmentation created")
            train_ds = train_ds.batch(self.batch_size)
            train_ds = train_ds.prefetch(self.AUTOTUNE)
            logging.info(f"Datasets corresponding to {self.sample_equal} created")
        else:
            train_dataset = self.make_ds(train_data['Image name'].values + '.jpg', train_data['Retinopathy grade'].values)
            train_ds = train_dataset.shuffle(600)#----------------------------------------------------------------------
            if self.augment:
                counter = tf.data.experimental.Counter()
                train_ds = tf.data.Dataset.zip((train_dataset, (counter, counter)))
                train_ds = train_ds.map(self.augment, num_parallel_calls = self.AUTOTUNE)
                logging.info(f"Datasets corresponding to {self.sample_equal} sampling and augmentation created")
            train_ds = train_ds.batch(self.batch_size)
            train_ds = train_ds.prefetch(self.AUTOTUNE)
            logging.info(f"Datasets corresponding to {self.sample_equal} created")


        return train_ds, test_ds, val_ds

    
    



