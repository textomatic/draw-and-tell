import time
import numpy as np
import pandas as pd
import pickle
import wikipediaapi as wapi
import openai


# Define constants
_ANIMAL_CLASS_FILE = 'animal_classes.txt'


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


def get_facts(url):
    """
    Sends an API request to OpenAI ChatGPT to obtain 10 interesting facts about the subject mentioned in given URL.

    Args:
        url(str): URL of a Wikipedia page

    Returns:
        (str): Content of the API response
    """
    prompt = f"Could you tell me 10 interesting facts about the main subject from the text contained in the following URL? {url}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']


def convert_to_df(animal_facts):
    """
    Converts the animal facts dictionary to a Pandas DataFrame for post-processing and returns a dictionary with same keys and the facts organized in a list for each key.
    
    Args:
        animal_facts(dict[str, str]): Dictionary containing animal names as keys and their raw string response of interesting facts obtained from ChatGPT as values
    
    Returns:
        animal_facts_clean_dict(dict[str, List(str)]): Dictionary containing animal names as keys and list of 10 interesting facts about them as values
    """
    # Place dictionary into a Pandas DataFrame
    df_animal_facts = pd.DataFrame({'animal': animal_facts.keys(), 'facts_raw': animal_facts.values()})

    # Remove newline characters, whitespaces, etc.
    df_animal_facts['facts_cleaned'] = df_animal_facts['facts_raw'].str.replace('\n\n', '\n')
    df_animal_facts['facts_cleaned'] = df_animal_facts['facts_cleaned'].str.replace('\d+[.]\s+', '', regex=True)
    df_animal_facts['facts_cleaned'] = df_animal_facts['facts_cleaned'].str.split('\n')

    # Remove first entry if length of list is more than 10, since it is a sentence about pleasantry and doesn't contain fact
    df_animal_facts['facts_cleaned'] = df_animal_facts['facts_cleaned'].apply(lambda x: x[1:] if len(x) > 10 else x)

    # Remove original raw string
    df_animal_facts_cleaned = df_animal_facts.drop('facts_raw', axis=1)

    # Convert dataframe back to dictionary
    animal_list = df_animal_facts_cleaned['animal'].tolist()
    facts_list = df_animal_facts_cleaned['facts_cleaned'].tolist()
    animal_facts_clean_dict = dict()
    for key, value in zip(animal_list, facts_list):
        animal_facts_clean_dict[key] = value
    
    return animal_facts_clean_dict


def main():
    # Load classes
    animal_list = load_classes(_ANIMAL_CLASS_FILE)

    # Initialize Wikipedia API object
    wiki = wapi.Wikipedia('en')

    # Configure OpenAI api key
    openai.api_key = "<your_api_key>"

    # Obtain Wikipedia page URL for each animal
    wiki_list = [wiki.page(animal) for animal in animal_list]

    # Pair them up with animal labels in a dict
    wiki_dict = dict(zip(animal_list, wiki_list))

    # Initialize an empty dict for storing facts
    animal_facts = dict()

    # Use a counter to pause for a while after every 3 API calls due to OpenAI restrictions
    counter = 0
    for key, val in wiki_dict.items():
        if val.exists() and animal_facts.get(key) is None:
            if counter == 3:
                print("Pausing momentarily to avoid rate limit error...")
                time.sleep(60)
                counter = 0
            print(f"Obtaining facts about {key}...")
            animal_facts[key] = get_facts(val.fullurl)
            counter += 1

    # Convert animal dict to dataframe for cleanup, and convert it back to dictionary for storage
    animal_facts_clean_dict = convert_to_df(animal_facts)

    # Save processed dictionary as pickled file
    with open('animal_facts_dict.pkl', 'wb') as f:
        pickle.dump(animal_facts_clean_dict, f)

    print("Animal facts successfully downloaded, processed, and saved!")


if __name__ == '__main__':
    main()