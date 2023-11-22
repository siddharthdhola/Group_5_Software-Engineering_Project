import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

# Downloading necessary resources for text processing
nltk.download('stopwords')
nltk.download('wordnet')

# Function to refine and standardize text for processing
def refine_text(input_text):
    # Remove any punctuation and convert text to lowercase
    cleaned_text = re.sub(r'[^\w\s]', '', input_text.lower())
    # Split text into words
    words_list = cleaned_text.split()
    # Filter out stopwords from the words list
    stop_words = stopwords.words('english')
    # Lemmatize each word to its base form
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_list if word not in stop_words]
    return ' '.join(lemmatized_words)

# Function to read JSON files and preprocess text data
def process_data_from_files(file_paths):
    processed_texts = []
    for path in file_paths:
        with open(path, 'r') as data_file:
            # Load data from JSON file
            data_content = json.load(data_file)
            # Iterate through nested data structure to extract prompts
            for source in data_content['Sources']:
                for sharing in source.get('ChatgptSharing', []):
                    for conversation in sharing.get('Conversations', []):
                        # Extract prompt and apply text refinement
                        prompt = conversation.get('Prompt', '')
                        cleaned_prompt = refine_text(prompt)
                        processed_texts.append(cleaned_prompt)
    return processed_texts

# Define paths to the JSON data files
file_names = ['discuss.json', 'issues.json', 'commit.json', 'pr_sharings.json','hn_sharings.json','file_sharing.json']

# Extract and preprocess data from the files
text_data = process_data_from_files(data_files)