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


# Define constants
_ANIMAL_CLASS_FILE = 'animal_classes.txt'
_DATA_PATH = '../data/train_simplified'
_IMAGE_SIZE = 224
_TRAIN_ROWS_PER_CLASS = 22500
_TEST_ROWS_PER_CLASS = 7500
_DATALOADER_BATCH_SIZE = 1024
_DATALOADER_NUM_WORKERS = 4
_NUM_EPOCHS = 15
_LEARNING_RATE = 0.0001
_MOMENTUM = 0.1
_MODEL_NAME = 'model_v1_svm'
_MODEL_CHECKPOINT = '../models/model_v1_svm.pth'
_PLOT_PATH = '../plots'


class SVM_Loss(torch.nn.modules.Module):
    """SVM loss function"""

    def __init__(self):
        """Initialize the SVM loss function"""
        super(SVM_Loss,self).__init__()

    def forward(self, outputs, labels, batch_size):
        """Forward pass of the SVM Loss function"""
        return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size


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

    
def save_plots(train_losses, train_accs, test_losses, test_accs, model_name):
    """
    Plots loss and accuracy of train and test phases and saves the plots as image files.

    Args:
        train_losses(np.ndarray): Numpy array containing train losses for all epochs
        train_accs(np.ndarray): Numpy array containing train accuracies for all epochs
        test_losses(np.ndarray): Numpy array containing test losses for all epochs
        test_accs(np.ndarray): Numpy array containing test accuracies for all epochs
        model_name(str): Name of the model, to be included in image file name

    Returns:
        None
    """
    # Plot accuracy vs epochs
    plt.figure(figsize=(10, 7))
    plt.plot(train_accs, color='green', linestyle='-', label='train accuracy')
    plt.plot(test_accs, color='blue', linestyle='-', label='test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{_PLOT_PATH}/accuracy_{model_name}.png")
    
    # Plot loss vs epochs
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color='orange', linestyle='-', label='train loss')
    plt.plot(test_losses, color='red', linestyle='-', label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{_PLOT_PATH}/loss_{model_name}.png")


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

    # Prepare data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5])
        ])
    }

    # Assign data transformations to train and test datasets    
    train_set.transforms = data_transforms['train']
    test_set.transforms = data_transforms['test']
    
    # Create dataloaders from train and test datasets
    train_loader = DataLoader(train_set, batch_size=_DATALOADER_BATCH_SIZE, shuffle=True, num_workers=_DATALOADER_NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=_DATALOADER_BATCH_SIZE, shuffle=False, num_workers=_DATALOADER_NUM_WORKERS)

    # Set up dictionary for dataloaders
    dataloaders = {'train': train_loader, 'test': test_loader}

    # Store size of training and test sets in a dictionary
    dataset_sizes = {'train': len(train_set), 'test': len(test_set)}

    return encoding_dict, dataloaders, dataset_sizes


def load_model(num_classes):
    """
    Loads a pre-trained PyTorch EfficientNet B3 model.

    Args:
        num_classes(int): Number of classes in the final linear layer

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

    return model


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, device):
    """
    Trains a neural network model with early stopping and saves best-performing model weights.

    Args:
        dataloaders(dict[str, torch.utils.data.DataLoader]): Dictionary containing the train and test dataloaders
        dataset_sizes(dict[str, int]): Dictionary containing the sizes of train and test datasets
        model(torch.nn.Module): A PyTorch neural network model
        criterion(torch.nn.<loss function class>): A PyTorch loss function
        optimizer(torch.optim.<optimization algorithm>): A PyTorch optimizer object
        device(str): Name of device, e.g. 'cuda', 'cpu'

    Returns:
        train_losses(np.ndarray): Numpy array containing train losses for all epochs
        train_accs(np.ndarray): Numpy array containing train accuracies for all epochs
        test_losses(np.ndarray): Numpy array containing test losses for all epochs
        test_accs(np.ndarray): Numpy array containing test accuracies for all epochs
    """
    # Define variables for early stopping
    early_stop_counter = 0
    early_stop_threshold = 3 # Number of epochs before early stopping is triggered
    early_stop_flag = False
    best_test_loss = np.inf # Best validation loss initialized to infinity
    best_test_acc = 0.0 # Best validation accuracy initialized to zero
    min_delta = 0.0001 # Tolerance of difference between validation loss and best validation loss
    
    # Initialize arrays for storing train and validation losses
    train_losses, train_accs, test_losses, test_accs = [], [], [], []    

    model.to(device) # Move model to device
    
    for i in tqdm(range(_NUM_EPOCHS), desc="Training progress"):
        print(f"Epoch {i+1}")
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            tloss, tcorrect = 0.0, 0 # Reset cumulative loss and correct count to 0 in each epoch

            for x, y in dataloaders[phase]:
                x, y = x.to(device), y.to(device)
                x = x.reshape(-1, _IMAGE_SIZE*_IMAGE_SIZE)
                optimizer.zero_grad() # Reset gradients for each minibatch

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(x)
                    _, preds = torch.max(output, 1) # Get the index of the highest scored class
                    loss = criterion(output, y, _DATALOADER_BATCH_SIZE)

                    if phase == 'train':
                        loss.backward() # Backpropagate loss to weights in network
                        optimizer.step() # Perform a step of gradient descent update

                tloss += loss.item() * x.size(0) # Update cumulative loss
                tcorrect += torch.sum(preds == y.data) # Update cumulative correct count

            epoch_loss = tloss / dataset_sizes[phase] # Calculate loss of this epoch
            epoch_acc = tcorrect.double() / dataset_sizes[phase] # Calculate accuracy of this epoch
            tqdm.write(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.4f}%')
            
            if phase == 'train': # Update arrays with train loss and accuracy
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.cpu())

            if phase == 'test': # Update arrays with validation loss and accuracy
                test_losses.append(epoch_loss)
                test_accs.append(epoch_acc.cpu())
                
                if epoch_loss < best_test_loss: # If epoch loss is better than best loss so far, update the latter
                    early_stop_counter = 0
                    best_test_loss = epoch_loss
                    best_test_acc = epoch_acc.cpu()
                    torch.save(model.state_dict(), _MODEL_CHECKPOINT) # Save model weights
                    print('Saved best model so far!')
                
                elif epoch_loss > (best_test_loss + min_delta): # If epoch loss is worse than best loss, trigger early stopping
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_threshold:
                        print('Model performance deterioriating consecutively in validation. Stopping early!')
                        early_stop_flag = True
                        break
        
        if early_stop_flag: # Break out of outer for loop
            break
        
        print()

    print(f"Best Validation Loss: {best_test_loss:.4f}")
    print(f'Best Validation Accuracy: {best_test_acc*100:.4f}%')

    # Convert lists to numpy arrays
    train_losses, train_accs, test_losses, test_accs = np.array(train_losses), np.array(train_accs), np.array(test_losses), np.array(test_accs)

    return train_losses, train_accs, test_losses, test_accs

    
def main():
    """
    Prepares the doodles dataset, trains a neural network model to classify labels of doodle, and evaluates the model performance.

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
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Test dataset size: {dataset_sizes['test']}")

    # Instantiate SVM regression model
    model = torch.nn.Linear(_IMAGE_SIZE*_IMAGE_SIZE, len(animal_list))
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define loss function
    criterion = SVM_Loss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE, momentum=_MOMENTUM)
    
    # Train model
    train_losses, train_accs, test_losses, test_accs = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, device)

    # Plot losses and accuracies
    save_plots(train_losses, train_accs, test_losses, test_accs, _MODEL_NAME)

    
if __name__ == '__main__':
    main()