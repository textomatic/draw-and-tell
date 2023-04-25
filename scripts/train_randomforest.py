import ast
import glob
import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, ConcatDataset
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# Define constants
_ANIMAL_CLASS_FILE = 'animal_classes.txt'
_DATA_PATH = '../data/train_simplified'
_IMAGE_SIZE = 64
_TRAIN_ROWS_PER_CLASS = 1000
_TEST_ROWS_PER_CLASS = 200


class DoodlesDataset(Dataset):
    """Custom class for the Quick! Draw It doodles dataset"""

    # Class variable of doodle image default size
    doodle_base_size = 256

    def __init__(self, csv_file, root_dir, encoding_dict, num_rows=1000, skip_rows=None, size=256, transform=None):
        """
        Args:
            csv_file(str): Path to the CSV file containing doodle data
            root_dir(str): Path to root directory containing all the doodles data files
            num_rows(int): Number of rows of the CSV file to read
            skip_rows(int): Number of rows to skip from the beginning of the CSV file
            size(int): Size of output image
            transform(torchvision.transforms): Torchvision transformations to be applied on a doodle, defaults to None
        """
        self.root_dir = root_dir
        self.size = size
        self.doodle = pd.read_csv(os.path.join(self.root_dir, csv_file), usecols=['drawing'], nrows=num_rows, skiprows=skip_rows)
        self.transform = transform
        self.label = get_label(encoding_dict, csv_file)


    @staticmethod
    def _draw(raw_strokes, size=256, line_weight=6, time_color=False):
        """
        Draws the doodle using OpenCV line function and resizes it to specified size if necessary.

        Args:
            raw_strokes(List[List[List[int], List[int]]]: Nested lists representing doodle strokes in sequence, where the outermost list represent a stroke, the first innermost list represents X displacement of each point, and the second innermost list represents Y displacement of each point
            size(int): Size of the doodle to be drawn
            line_weight(int): Size of the line
            time_color(bool): Shows stroke of varying color as a function of time, defaults to False

        Returns:
            img(np.ndarray): Numpy array representing the doodle
        """
        # Initialize doodle as empty numpy matrix
        img = np.zeros((DoodlesDataset.doodle_base_size, DoodlesDataset.doodle_base_size), np.uint8)

        # Draw doodle according to sequence of strokes
        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if time_color else 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, line_weight)
        
        # Resize doodle if applicable
        if size != DoodlesDataset.doodle_base_size:
            img = cv2.resize(img, (size, size))
        
        return img
    

    def __len__(self):
        """Returns size of the dataset"""
        return len(self.doodle)


    def __getitem__(self, index):
        """
        Gets one doodle and its label by index.

        Args:
            index(int): Index representing target row in doodle dataset to access
        
        Returns:        
            (np.ndarray): Numpy array representing the doodle
            (int): Numerically encoded label of the doodle
        """
        # Perform literal evaluation of string to obtain raw strokes
        raw_strokes = ast.literal_eval(self.doodle.drawing[index])

        # Call static draw method to get doodle drawn
        doodle = self._draw(raw_strokes, size=self.size, line_weight=2)

        # Apply transformations on doodle if applicable
        if self.transform:
            doodle = self.transform(doodle)
        
        return (doodle[None]/255).astype('float32'), self.label


def decode_labels(dec_dict, label):
    """
    Obtains the class label (string) represented by the given numerically encoded label.

    Args:
        dec_dict(dict[int, str]): Dictionary containing the numerically encoded label and their corresponding class label
        label(int): Numerical encoded label 

    Returns:
        (str): Class label corresponding to the encoded label
    """
    return dec_dict[label]


def get_label(encoding_dict, filename):
    """
    Obtains the numerically encoded label of a class given a CSV file of doodle data.

    Args:
        encoding_dict(dict[str, int]): Dictionary containing the class label and their corresponding numerically encoded label
        filename(str): Path to a CSV file containing doodle data
    
    Returns:
        (int): Numerically encoded label of a class
    """
    return encoding_dict[filename[:-4].split('/')[-1].replace(' ', '_')]


def load_classes(filename):
    """
    Loads the class labels from a text file.
    
    Args:
        filename(str): Path to the text file containing class labels

    Returns:
        classes(List[str]): List containing all class labels
    """
    with open(filename, 'r') as f:
        classes = f.read()
    return classes.split("\n")


def prepare_data(animal_list):
    """
    Prepares dataset as dataloaders and adds transformations to them.

    Args:
        animal_list(List[str]): List containing all class labels

    Returns:
        encoding_dict(dict[str, int]): Dictionary containing the class label and their corresponding encoding number 
        dataloaders(dict[str, torch.utils.data.DataLoader]): Dictionary containing the train and test dataloaders
        dataset_sizes(dict[str, int]): Dictionary containing the sizes of train and test datasets
    """
    # Get all the CSV files containing doodle data
    filenames = glob.glob(os.path.join(_DATA_PATH, '*.csv'))

    # Filter classes to only those in animal_list
    filtered_filenames = []
    for file in filenames:
        if file.split('/')[-1].split('.')[0].replace(' ', '_') in animal_list:
            filtered_filenames.append(file)
    
    # Initialize dictionary to encode label with numbers
    encoding_dict = {}
    counter = 0
    for fn in filtered_filenames:
        encoding_dict[fn[:-4].split('/')[-1].replace(' ', '_')] = counter
        counter += 1
    
    # Create an inverted dictionary for decoding number to label
    decoding_dict = {v: k for k , v in encoding_dict.items()}
    
    # Create train set by concatenating dataset from all classes
    train_set = ConcatDataset([DoodlesDataset(fn.split('/')[-1], _DATA_PATH, encoding_dict, num_rows=_TRAIN_ROWS_PER_CLASS, size=_IMAGE_SIZE) for fn in filtered_filenames])
    
    # Create test set by concatenating dataset from all classes
    test_set = ConcatDataset([DoodlesDataset(fn.split('/')[-1], _DATA_PATH, encoding_dict, num_rows=_TEST_ROWS_PER_CLASS, size=_IMAGE_SIZE, skip_rows=range(1, _TRAIN_ROWS_PER_CLASS+1)) for fn in filtered_filenames])

    train_data, train_labels, test_data, test_labels = [], [], [], []

    # Convert train data to numpy array to be used in Scikit-Learn
    for data in train_set:
        inputs, labels = data
        images = inputs.reshape(-1)
        train_data.append(images)
        train_labels.append([labels])

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Convert test data to numpy array to be used in Scikit-Learn
    for data in test_set:
        inputs, labels = data
        images = inputs.reshape(-1)
        test_data.append(images)
        test_labels.append([labels])

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


def train_model(train_data, train_labels, test_data, test_labels):
    """
    Trains a Random Forest Classifier and evaluates its performance.

    Args:
        train_data(np.ndarray): Train data
        train_labels(np.ndarray): Train labels
        test_data(np.ndarray): Test data
        test_labels(np.ndarray): Test labels

    Returns:
        None
    """
    model = RandomForestClassifier(random_state=45)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy}")


def main():
    # Load classes
    animal_list = load_classes(_ANIMAL_CLASS_FILE)

    # Prepare data
    train_data, train_labels, test_data, test_labels = prepare_data(animal_list)

    # Train the model
    train_model(train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    main()