import torch
import torchvision
import coremltools as ct


# Define constants
_ANIMAL_CLASS_FILE = 'animal_classes.txt'
_TORCH_MODEL_WEIGHTS = '../models/model_v11_efficientnetb3.pth'
_COREML_MODEL_NAME = '../models/drawandtell_v1.mlmodel'
_COREML_MODEL_AUTHOR = "Shen Juin Lee"
_COREML_MODEL_DESC = "Predicts the animal depicted in a doodle."


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


def load_trained_model(model_weights, num_classes):
    """
    Loads a trained PyTorch EfficientNet B3 model.

    Args:
        model_weights(str): Path to the pickled state_dict of a PyTorch EfficientNet B3 model containing its trained weights
        num_classes(int): Number of classes in the final linear layer

    Returns:
        torch_model(torch.nn.Module): A PyTorch EfficientNet B3 model
    """
    # Load pre-trained model with default weights
    torch_model = torchvision.models.efficientnet_b3(weights='DEFAULT')

    # Change number of input channel to 1 and squeeze first layer weights
    torch_model.features[0][0].in_channels = 1
    torch_model.features[0][0].weight.data = torch_model.features[0][0].weight.data.sum(dim=1)[:,None]

    # Change the number of output classes in final layer
    fc_in_features = torch_model.classifier[1].in_features
    torch_model.classifier[1] = torch.nn.Linear(in_features=fc_in_features, out_features=num_classes)

    # Load weights from trained model
    torch_model.load_state_dict(torch.load(model_weights))

    # Add softmax layer to ensure converted CoreML model will have Softmax too
    torch_model = torch.nn.Sequential(
        torch_model,
        torch.nn.Softmax(dim=1)
    )

    return torch_model


def convert_torch_to_coreml(animal_list, torch_model, author, short_desc, device):
    """
    Converts PyTorch model to CoreML model.

    Args:
        animal_list(List[str]): List containing all class labels
        torch_model(torch.nn.Module): A PyTorch neural network model
        author(str): Author of the model, as part of the CoreML model metadata
        short_desc(str): Short description of the model, as part of the CoreML model metadata
        device(str): Name of device, e.g. 'cuda', 'cpu'

    Returns:
        model(coremltools.models.model.MLModel): The converted CoreML model
    """
    # Shift model to device
    torch_model.to(device)

    # Switch model to evaluation mode
    torch_model.eval()

    # Trace the model with random data.
    example_input = torch.rand(1, 1, 224, 224)
    example_input = example_input.to(device)
    traced_model = torch.jit.trace(torch_model, example_input)

    # Scale the input by 1/255 as image preprocessing
    scale = 1/255

    # Define ImageType input and select grayscale as color since images are single channel only
    image_input = ct.ImageType(name="image_input",
                           shape=example_input.shape,
                           scale=scale,
                           color_layout=ct.colorlayout.GRAYSCALE,
                          channel_first=None)
    
    # Convert to Core ML using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        inputs=[image_input],
        classifier_config = ct.ClassifierConfig(animal_list),
        compute_units=ct.ComputeUnit.ALL
    )

    # Set the model metadata
    model.author = author
    model.short_description = short_desc

    return model


def main():
    """
    Converts PyTorch model to CoreML model for deployment on iOS.

    Args:
        None

    Returns:
        None
    """
    # Load animal classes
    animal_list = load_classes(_ANIMAL_CLASS_FILE)

    # Load trained model
    torch_model = load_trained_model(_TORCH_MODEL_WEIGHTS, len(animal_list))

    # Check if GPU device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert PyTorch model to CoreML model
    model = convert_torch_to_coreml(animal_list, torch_model, _COREML_MODEL_AUTHOR, _COREML_MODEL_DESC, device)

    # Save the converted model.
    model.save(_COREML_MODEL_NAME)

    # Print a confirmation message.
    print('Model converted and saved!')


if __name__ == '__main__':
    main()