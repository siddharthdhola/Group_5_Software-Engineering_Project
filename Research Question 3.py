import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Function to standardize and preprocess the text
def preprocess_text(raw_text):
    raw_text = re.sub(r'[^\w\s]', '', raw_text.lower())
    tokenized_words = raw_text.split()
    filtered_words = [word for word in tokenized_words if word not in stopwords.words('english')]
    word_lemmatizer = WordNetLemmatizer()
    lemmatized_words = [word_lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

# Function to load and process data from JSON files
def process_data(file_paths):
    processed_data = []
    for path in file_paths:
        with open(path, 'r') as data_file:
            json_data = json.load(data_file)
            for source_item in json_data['Sources']:
                for sharing in source_item.get('ChatgptSharing', []):
                    for dialogue in sharing.get('Conversations', []):
                        prompt_text = dialogue.get('Prompt', '')
                        standardized_text = preprocess_text(prompt_text)
                        processed_data.append(standardized_text)
    return processed_data

# File paths
json_file_paths = ['discuss.json', 'issues.json', 'commit.json', 'pr_sharings.json','hn_sharings.json','file_sharing.json']

# Extract and preprocess the data
text_data = process_data(json_file_paths)

# Text to TF-IDF matrix conversion
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

# Applying K-Means clustering
cluster_count = 4
kmeans_model = KMeans(n_clusters=cluster_count, random_state=42)
kmeans_model.fit(tfidf_matrix)

# Extracting cluster labels
labels = kmeans_model.labels_

# Mapping labels to categories
categories = {
    0: 'Content Generation',
    1: 'Information Retrieval',
    2: 'Natural language understanding',
    3: 'Language Translation'
}

# Counting labels
category_counts = {label: list(labels).count(label) for label in set(labels)}

print("Category counts with names:")
for label, count in category_counts.items():
    print(f"{categories[label]}: {count}")


# Plotting the distribution of categories
plt.figure(figsize=(8, 6))
plt.bar(categories.values(), category_counts.values(), color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Prompt Counts')
plt.title('Distribution of Prompt Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()