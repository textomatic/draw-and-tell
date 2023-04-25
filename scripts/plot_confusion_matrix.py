import ast
import glob
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Define constants
_ANIMAL_CLASS_FILE = 'animal_classes.txt'
_DATA_PATH = '../data/train_simplified'
_IMAGE_SIZE = 224
_TRAIN_ROWS_PER_CLASS = 22500
_TEST_ROWS_PER_CLASS = 7500
_DATALOADER_BATCH_SIZE = 256
_DATALOADER_NUM_WORKERS = 4
_MODEL_NAME = 'model_v11_efficientnetb3'
_MODEL_WEIGHTS = '../models/model_v11_efficientnetb3.pth'
_PLOT_PATH = '../plots'


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

    # Create test set by concatenating dataset from all classes
    test_set = ConcatDataset([DoodlesDataset(fn.split('/')[-1], _DATA_PATH, encoding_dict, num_rows=_TEST_ROWS_PER_CLASS, size=_IMAGE_SIZE, skip_rows=range(1, _TRAIN_ROWS_PER_CLASS+1)) for fn in filtered_filenames])

    # Prepare data transformations
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5])
        ])
    }

    # Assign data transformations to test dataset 
    test_set.transforms = data_transforms['test']
    
    # Create dataloaders from test dataset
    test_loader = DataLoader(test_set, batch_size=_DATALOADER_BATCH_SIZE, shuffle=False, num_workers=_DATALOADER_NUM_WORKERS)

    # Set up dictionary for test dataloader
    dataloaders = {'test': test_loader}

    # Store size of test set in a dictionary
    dataset_sizes = {'test': len(test_set)}

    return encoding_dict, decoding_dict, dataloaders, dataset_sizes


def load_model(model_weights, num_classes, device):
    """
    Loads a trained PyTorch EfficientNet B3 model.

    Args:
        model_weights(str): Path to the pickled state_dict of a PyTorch EfficientNet B3 model containing its trained weights
        num_classes(int): Number of classes in the final linear layer
        device(str):  Name of device where model weights reside, e.g. 'cuda', 'cpu'

    Returns:
        torch_model(torch.nn.Module): A PyTorch EfficientNet B3 model
    """
    # Load pre-trained EfficientNet B3 model
    model = torchvision.models.efficientnet_b3(weights='DEFAULT')

    # Change number of input channel to 1 and squeeze first layer weights
    model.features[0][0].in_channels = 1
    model.features[0][0].weight.data = model.features[0][0].weight.data.sum(dim=1)[:,None]

    # Change the number of output classes in final layer
    fc_in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features=fc_in_features, out_features=num_classes)

    # Load weights from trained model
    model.load_state_dict(torch.load(model_weights, map_location=torch.device(device)))

    return model


def get_predictions(model, dataloaders, device):
    """
    Passes test dataset inputs through a trained model and returns the truth labels and predicted labels.

    Args:
        model(torch.nn.Module): A PyTorch EfficientNet B3 model
        dataloaders(dict[str, torch.utils.data.DataLoader]): Dictionary containing the test dataloader
        device(str): Name of device, e.g. 'cuda', 'cpu'
    
    Returns:
        test_y(List[int]): List containing truth labels of test dataset
        test_preds(List[int]): List containing predicted labels of test dataset
    """
    with torch.no_grad():
        model.eval()
        test_y = []
        test_preds = []

        for x, y in tqdm(dataloaders['test']):
            x, y = x.to(device), y.to(device)
            output = model(x)
            # Convert raw outputs to probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(), axis=1)
            # Add preds and truths to lists
            test_preds.extend(preds)
            test_y.extend(y.cpu())
    
    return test_y, test_preds


def main():
    """
    Evaluates a trained model and plots its confusion matrix.

    Args:
        None

    Returns:
        None
    """
    # Load classes
    animal_list = load_classes(_ANIMAL_CLASS_FILE)
    
    # Prepare data
    encoding_dict, dataloaders, dataset_sizes = prepare_data(animal_list)

    # Print out size of train and test datasets
    print(f"Test dataset size: {dataset_sizes['test']}")

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model
    model = load_model(_MODEL_WEIGHTS, len(animal_list), device)
    
    # Move model to device
    model.to(device)

    # Evaluate model with test dataset
    test_y, test_preds = get_predictions(model, dataloaders, device)

    # Build confusion matrix and store in a Pandas DataFrame
    cf_matrix = confusion_matrix(test_y, test_preds)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix, axis=1)[:, None], index=[i for i in animal_list], columns=[i for i in animal_list])

    # Plot heatmap of confusion matrix and save plot
    plt.figure(figsize = (100,100))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(f'{_PLOT_PATH}/confusion_matrix_{_MODEL_NAME}.png')
    

if __name__ == '__main__':
    main()