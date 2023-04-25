import os
from io import BytesIO
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torchvision
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from gtts import gTTS


# Define constants
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_ANIMAL_CLASS_PATH = os.path.join(_CURRENT_DIR, 'assets/animal_classes.txt')
_ANIMAL_FACTS_PATH = os.path.join(_CURRENT_DIR, 'assets/animal_facts_dict.pkl')
_TORCH_MODEL_WEIGHTS = os.path.join(_CURRENT_DIR, 'assets/model_v11_efficientnetb3.pth')


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


def predict_doodle(doodle_tensor, animal_list, model, device):
    """
    Predicts classes of doodle and returns the top 3 highest-scored results.

    Args:
        doodle_tensor(torch.Tensor): PyTorch tensor containing the doodle
        animal_list(List[str]): List containing all class labels
        model(torch.nn.Module): A PyTorch neural network model
        device(str): Name of device, e.g. 'cuda', 'cpu'
    
    Returns
        (List[tuple(str, float)]): List of tuples consisting of class label and confidence score pairs
    """
    with torch.no_grad():
        model.eval()
        pred_image = doodle_tensor.to(device)
        # Get predictions
        logits = model(pred_image)
        # Obtain softmax distribution of predictions
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    # Create array of class label and prediction score pairs
    results = [(i, j) for i, j in zip(animal_list, list(probs[0]))]
    
    # Sort array in descending order of prediction score
    results = sorted(results, key=lambda item: item[1], reverse=True)

    # Return top 3 results
    return results[:3]


def main():
    # Load animal facts dictionary from pickled file
    with open(_ANIMAL_FACTS_PATH, 'rb') as f:
        animal_facts_dict = pickle.load(f)

    # Load animal classes
    animal_list = load_classes(_ANIMAL_CLASS_PATH)

    # Check if GPU device is available
    device = 'cpu' if torch.cuda.is_available() else 'cpu'

    # Load trained model onto device
    model = load_model(_TORCH_MODEL_WEIGHTS, len(animal_list), device)
    model.to(device)

    # Setup streamlit page configs
    st.set_page_config(page_title='Draw and Tell ✏️')
    st.title('Draw and Tell ✏️')
    st.markdown('Draw an animal and let us guess what it is!')

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="black",
        background_color="#eee",
        background_image=None,
        update_streamlit=True,
        width=450,
        height=450,
        drawing_mode='freedraw',
        point_display_radius=0,
        display_toolbar=True,
        key="doodle_canvas",
    )
    
    # Create button for triggering classification
    guess_button = st.button('Guess')
    
    # Checkbox to display confidence score of predictions
    display_confidence = st.checkbox("Show Confidence", False)
    
    st.markdown("")
    st.markdown("")
    
    # Proceed if button is pressed
    if guess_button:
        # If canvas is not empty, perform inference
        if canvas_result.json_data["objects"]:
            with st.spinner('Guessing...'):
                doodle = canvas_result.image_data
                doodle = doodle[:,:,:1]
                # Convert the doodle image to Torch tensor
                doodle_tensor = torch.from_numpy(doodle).to(torch.float32)/255
                # Add additional dimension for batch
                doodle_tensor = doodle_tensor[None]
                # Ensure input dimensions are correct for Torch model
                doodle_tensor = torch.permute(doodle_tensor, (0, 3, 1, 2))
                # Get predictions
                results = predict_doodle(doodle_tensor, animal_list, model, device)

                # Display confidence scores if selected
                if display_confidence:
                    st.subheader("Top 3 Guesses:")
                    for result in results:
                        st.markdown("**" + f"{result[0].capitalize()}" + "**, confidence: " + "`" + f"{result[1]*100:.2f}%" + "`")
                else:
                    st.subheader("Top Guess:")
                    st.markdown(f"{results[0][0].capitalize()}")
                    
                    # Display a fun fact about the top scored label
                    fun_fact = animal_facts_dict[results[0][0]][np.random.randint(0, 10)]
                    st.subheader("Fun Fact:")
                    st.markdown(f"{fun_fact}")
                    
                    # Use Google Translate's Text-to-Speech API to read out the fun fact
                    ff_sound_file = BytesIO()
                    tts = gTTS(fun_fact, lang='en', slow=True)
                    tts.write_to_fp(ff_sound_file)
                    
                    st.markdown('_Listen to the fun fact:_ 🔊')
                    st.audio(ff_sound_file)
        
        else:
            # Display message if nothing is drawn on canvas
            st.info("Canvas is empty! Please draw something and try again")


if __name__ == '__main__':
    main()