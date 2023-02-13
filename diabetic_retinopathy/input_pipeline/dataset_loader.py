import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.image import ResizeMethod
import tensorflow_addons as tfa
import numpy as np
import cv2
import pandas as pd
import logging
import tensorflow as tf
import pathlib
import gin

PRINT_DATASET_DISTRIBUTIONS = False


@gin.configurable
class DatasetLoader:
    """
    Class that loads and processes the dataset.
    
    Args:
        dataset_name (str): Name of the dataset to be loaded. 
                            Accepted dataset names: 'idrid', 'eyepacs', 'mnist'.
        dataset_directory (str): Path to the directory where the dataset is stored.
        selected_model (str): Model architecture to be used. 
                              Accepted model names: 'baseline', 'INception_v3', 'resnet_50', 'efficient', 'none'
        TRAINING_DATASET_RATIO (float): Ratio of the training dataset. 
                                        0.0 < ratio <= 1.0
        SAMPLE_EQUAL (bool): If true, the class distribution in the training dataset is made equal.
        AUGMENT (bool): If true, the data is augmented.
        BATCH_SIZE (int): Batch size. Preferably a power of 2.
        EQUALIZATION (bool): If true, histogram EQUALIZATION is performed on the images.
        output_dataset_directory (str): Path to the directory where the processed dataset is stored.
    """

    def __init__(self, dataset_name, dataset_directory, selected_model,
                 TRAINING_DATASET_RATIO, SAMPLE_EQUAL, AUGMENT, BATCH_SIZE, EQUALIZATION,
                 output_dataset_directory):

        self.selected_model = selected_model
        if self.selected_model.lower() not in ['baseline', 'inception_v3', 'resnet_50', 'efficient', 'none']:
            raise ValueError(f"{self.selected_model} not available")

        if BATCH_SIZE > 0 and isinstance(BATCH_SIZE, int):
            self.BATCH_SIZE = BATCH_SIZE
        else:
            raise ValueError(
                f"Received batch size {BATCH_SIZE}. Accepted size > 0. Preferrably in powers of 2")

        self.dataset_name = dataset_name
        accepted_dataset_names = ('idrid', 'eyepacs', 'mnist')
        if self.dataset_name not in accepted_dataset_names:
            raise ValueError(
                f"Received invalid dataset name: '{self.dataset_name}', accepted dataset names: {accepted_dataset_names}")

        self.dataset_directory = pathlib.Path(dataset_directory)
        if (not self.dataset_directory.exists()) or (not self.dataset_directory.is_dir()):
            raise ValueError(
                f"Received invalid dataset directory: '{self.dataset_directory}'.")

        self.output_dataset_directory = pathlib.Path(output_dataset_directory)
        if (not self.output_dataset_directory.exists()) or (not self.output_dataset_directory.is_dir()):
            raise ValueError(
                f"Received invalid output dataset directory: '{self.output_dataset_directory}'.")

        if isinstance(TRAINING_DATASET_RATIO, float) and (0.0 < TRAINING_DATASET_RATIO <= 1.0):
            self.TRAINING_DATASET_RATIO = TRAINING_DATASET_RATIO
        else:
            raise ValueError(
                f'The training dataset split ratio has to be: 0.0 < ratio <= 1.0. Received ratio: {TRAINING_DATASET_RATIO}')

        if isinstance(SAMPLE_EQUAL, bool):
            self.SAMPLE_EQUAL = SAMPLE_EQUAL
        else:
            raise ValueError(
                f"(Required True or False. Received {SAMPLE_EQUAL})")

        if isinstance(AUGMENT, bool):
            self.augment_needed = AUGMENT
        else:
            raise ValueError(f"(Required True or False. Received {AUGMENT})")

        if isinstance(EQUALIZATION, bool):
            self.EQUALIZATION = EQUALIZATION
        else:
            raise ValueError(
                f"(Required True or False. Received {EQUALIZATION})")

        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def parse_function(self, filename, label):
        """
        This function takes in two arguments: filename and label.
        It reads the image file, decodes it, preprocesses it based on the selected model, crops it, resizes it and returns the processed image and its label.
        """
        image_string = tf.io.read_file(str(self.dataset_directory) + filename)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.io.decode_jpeg(image_string, channels=3)

        if self.selected_model.lower() == 'baseline':
            # This will convert to float values in [0, 1]
            image = tf.image.convert_image_dtype(image, tf.float32)
        elif self.selected_model.lower() == 'inception_v3' or self.selected_model.lower() == 'resnet_50':
            image = tf.cast(image, tf.float32)
            if self.selected_model.lower() == 'inception_v3':
                # This will convert to float values in [-1,1]
                image = tf.keras.applications.inception_v3.preprocess_input(
                    image)
            elif self.selected_model.lower() == 'resnet_50':
                # This will convert to float values in [-1,1]
                image = tf.keras.applications.resnet_v2.preprocess_input(image)

        # Creating a bounding box to get more focus on the image
        image = tf.image.crop_to_bounding_box(image, 0, 200, 2848, 3550)

        image = tf.image.resize_with_pad(
            image, 256, 256, method=ResizeMethod.BILINEAR, antialias=False)
        return image, label

    def make_ds(self, images_paths, labels):
        """
        This function takes in two arguments: images_paths and labels.
        It creates a dataset from the given image paths and labels, shuffles it, maps it using parse_function, and returns the processed dataset.
        """
        ds = tf.data.Dataset.from_tensor_slices((images_paths, labels))
        ds = ds.shuffle(5000)
        ds = ds.map(self.parse_function, num_parallel_calls=self.AUTOTUNE)
        return ds

    def create_datasets(self):
        """
        This function creates datasets depending on the selected dataset name and returns the train, test, and validation datasets.
        """
        if self.dataset_name == 'idrid':
            logging.info(f"Preparing dataset '{self.dataset_name}'...")
            if self.EQUALIZATION:
                self._save_image()
            train_ds, test_ds, valid_ds = self._create_idrid_dataset()
        elif self.dataset_name == 'eyepacs':
            logging.info(
                "Sampling is not implemented for the 'eyepacs' dataset.")
        elif self.dataset_name == 'mnist':
            logging.info(
                "Sampling is not implemented for the 'MNIST' dataset.")

        return train_ds, test_ds, valid_ds

    def clahe_processed_dataset_gen(self, input_dir, output_dir):
        """
        This function applies CLAHE to images in a given input directory and saves the processed images in a given output directory.

        Parameters:
            input_dir (str): The path to the directory containing the images to be processed.
            output_dir (str): The path to the directory where the processed images will be saved.
        """

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
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                img_lab[..., 0] = clahe.apply(img_lab[..., 0])
                img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)

                # Save the processed image
                cv2.imwrite(os.path.join(output_dir, filename), img_clahe)

    def _save_image(self):
        """
        This function saves the processed images in the output directory. It first applies CLAHE processing to the training images in the input directory if the output directory is empty or if the number of images in the output directory is less than 413. Then it applies CLAHE processing to the test images in the input directory if the output directory is empty or if the number of images in the output directory is less than     103.

        Returns:
            None
        """
        input_dir = str(self.dataset_directory)
        output_dir = str(self.output_dataset_directory)
        input_train_path = input_dir+"/images/train"
        input_test_path = input_dir+"/images/test"
        output_train_path = output_dir+"/images/train"
        output_test_path = output_dir+"/images/test"
        if not os.path.exists(output_train_path):
            self.clahe_processed_dataset_gen(
                input_train_path, output_train_path)
        elif len(os.listdir(output_train_path)) < 413 :
            self.clahe_processed_dataset_gen(
                input_train_path, output_train_path)
        logging.info(f"training clahe done")
        if not os.path.exists(output_test_path):
            self.clahe_processed_dataset_gen(input_test_path, output_test_path)
        elif len(os.listdir(output_test_path)) < 103:
            self.clahe_processed_dataset_gen(input_test_path, output_test_path)
        logging.info(f"testing clahe done")

    def augment(self, images_labels, seed):
        """
        A function to augment an image using various techniques like flipping, rotating and changing brightness/contrast/saturation.

        Parameters:
            images_labels (tuple): A tuple containing the images and their corresponding labels
            seed (int): Random seed for the image augmentation

        Returns:
            tuple: A tuple containing the augmented images and their labels

        """
        images, labels = images_labels
        # Make a new seed.
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        images = tf.image.stateless_random_flip_left_right(
            images, seed=new_seed)
        k = np.random.uniform(low=1.0, high=20.0)
        images = tfa.image.rotate(images, tf.constant(np.pi / k))
        images = tf.image.stateless_random_flip_up_down(images, seed=new_seed)
        k = np.random.uniform(low=1.0, high=20.0)
        images = tfa.image.rotate(images, 20)
        # images = tf.image.stateless_random_contrast(images , 0.2, 0.5, seed = new_seed)
        # images = tf.image.stateless_random_brightness(images, 0.2, seed = new_seed)
        # images = tf.image.stateless_random_saturation(images, lower=0.1, upper=0.9, seed= new_seed)
        # images = tf.image.stateless_random_jpeg_quality(images, 60, 90, seed = new_seed)
        return images, labels

    def _create_idrid_dataset(self):
        '''Read image and label files from data directory.

        Returns
        -------
        tuple of two tf.data.Dataset
            The training and test dataset, the training set is not yet split.
        '''
        logging.info(
            f"Preparing file loading from {self.dataset_directory}...")

        # Parse the CSV files
        labels_base_path = str(self.dataset_directory) + '/labels'
        train_label_data_dir = labels_base_path + "/train.csv"
        test_label_data_dir = labels_base_path + "/test.csv"
        # logging.info(test_label_data_dir)

        column_names = ["Image name",
                        "Retinopathy grade", "Risk of macular edema"]
        training_data = pd.read_csv(
            train_label_data_dir, names=column_names, usecols=column_names)
        # Removing the first row as it contains header
        training_data = training_data.iloc[1:]
        # Changing from multiclass to binary class
        training_data['Retinopathy grade'].replace(
            {'0': 0, '1': 0, '2': 1, '3': 1, '4': 1}, inplace=True)

        test_data = pd.read_csv(test_label_data_dir,
                                names=column_names, usecols=column_names)
        # Removing the first row as it contains header
        test_data = test_data.iloc[1:]
        # Changing from multiclass to binary class
        test_data['Retinopathy grade'].replace(
            {'0': 0, '1': 0, '2': 1, '3': 1, '4': 1}, inplace=True)
        # logging.info(test_data.head())
        # logging.info(f"Conversion completed")
        total_data_amount = len(training_data)

        train_data = training_data.loc[:int(
            self.TRAINING_DATASET_RATIO * total_data_amount)]
        val_data = training_data.loc[int(
            self.TRAINING_DATASET_RATIO * total_data_amount) + 1:]

        # Creating pos and neg train dataset for later balancing
        pos_train_data = train_data[train_data['Retinopathy grade'] == 1]
        neg_train_data = train_data[train_data['Retinopathy grade'] == 0]
        # logging.info(f"Positive and negative datasets created")
        # logging.info(pos_train_data.head())

        pos_images_paths = '/images/train/' + \
            pos_train_data['Image name'].values + '.jpg'
        neg_images_paths = '/images/train/' + \
            neg_train_data['Image name'].values + '.jpg'
        val_images_path = '/images/train/' + \
            val_data['Image name'].values + '.jpg'
        # logging.info(f"paths found")
        # logging.info(pos_images_paths.head())

        pos_labels = pos_train_data['Retinopathy grade'].values
        neg_labels = neg_train_data['Retinopathy grade'].values
        val_labels = val_data['Retinopathy grade'].values
        # logging.info(f"Values known")
        # logging.info(pos_labels)

        if self.EQUALIZATION:
            self.dataset_directory = self.output_dataset_directory
        pos_ds = self.make_ds(pos_images_paths, pos_labels)
        neg_ds = self.make_ds(neg_images_paths, neg_labels)
        val_ds = self.make_ds(
            val_images_path, val_labels).batch(self.BATCH_SIZE)

        test_images_path = '/images/test/' + \
            test_data['Image name'].values + '.jpg'
        test_labels = test_data['Retinopathy grade'].values
        test_ds = self.make_ds(
            test_images_path, test_labels).batch(len(test_labels))
        # logging.info(f"All basic datasets created")

        # Oversampling neg_ds to pos_ds
        sample_neg_ds = neg_ds.take(len(pos_ds) - len(neg_ds))
        neg_ds = neg_ds.concatenate(sample_neg_ds)

        if self.SAMPLE_EQUAL:
            train_dataset = tf.data.Dataset.sample_from_datasets(
                [pos_ds, neg_ds], weights=[0.5, 0.5])
            train_ds = train_dataset.shuffle(600)
            if self.augment:
                counter = tf.data.experimental.Counter()
                train_ds = tf.data.Dataset.zip(
                    (train_dataset, (counter, counter)))
                train_ds = train_ds.map(
                    self.augment, num_parallel_calls=self.AUTOTUNE)
                logging.info(
                    f"Datasets corresponding to {self.SAMPLE_EQUAL} sampling and augmentation created")
            train_ds = train_ds.batch(self.BATCH_SIZE)
            train_ds = train_ds.prefetch(self.AUTOTUNE)
            logging.info(
                f"Datasets corresponding to {self.SAMPLE_EQUAL} created")
        else:
            train_dataset = tf.data.Dataset.sample_from_datasets(
                [pos_ds, neg_ds], weights=[0.62, 0.38])
            train_ds = train_dataset.shuffle(600)
            if self.augment:
                counter = tf.data.experimental.Counter()
                train_ds = tf.data.Dataset.zip(
                    (train_dataset, (counter, counter)))
                train_ds = train_ds.map(
                    self.augment, num_parallel_calls=self.AUTOTUNE)
                logging.info(
                    f"Datasets corresponding to {self.SAMPLE_EQUAL} sampling and augmentation created")
            train_ds = train_ds.batch(self.BATCH_SIZE)
            train_ds = train_ds.prefetch(self.AUTOTUNE)
            logging.info(
                f"Datasets corresponding to {self.SAMPLE_EQUAL} created")

        return train_ds, test_ds, val_ds
