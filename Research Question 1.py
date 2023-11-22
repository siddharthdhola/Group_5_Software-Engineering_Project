import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Loadingw NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

labelling_keys = {
    'bug': ['bug', 'error', 'issue', 'glitch', 'malfunction', 'fault', 'defect', 'failure', 'anomaly', 'problem',
            'flaw', 'error message', 'crash', 'hang', 'freeze', 'unexpected behavior', 'non-responsive',
            'runtime error', 'not working', 'broken', 'wrong', 'unresponsive', 'slow', 'stuck', 'hangs', 'crashing',
            'error code', 'freezing', 'lag', 'blank screen', 'disconnects', 'fails', 'corruption', 'data loss',
            'inconsistency', 'interruption', 'misbehavior', 'performance issue', 'sync problem', 'bug report',
            'code fix', 'debugging', 'program error', 'system crash', 'application error', 'software issue',
            'coding mistake', 'logic error', 'exception', 'instability', 'incorrect output', 'infinite loop',
            'memory leak', 'stack trace', 'segmentation fault', 'compiler error', 'fatal error', 'kernel panic',
            'core dumped', 'pytest failed', 'build failed'],
    'feature request': ['feature request', 'enhancement', 'improvement', 'addition', 'upgrade', 'new functionality',
                        'enhanced capability', 'extension', 'modification', 'update request', 'new feature',
                        'improvement suggestion', 'expansion', 'development request', 'optimization', 'refinement',
                        'customization', 'user experience improvement', 'want', 'need', 'add', 'wish', 'suggest',
                        'hope for', 'could use', 'should have', 'requesting', 'looking for', 'asking for', 'desire',
                        'would like', 'prefer', 'would be nice', 'integration', 'usability', 'accessibility',
                        'compatibility', 'automation', 'scalability', 'user story', 'roadmap item', 'new capability',
                        'additional functionality', 'enhanced performance', 'simplification', 'modernization',
                        'better interface', 'personalization', 'configurability', 'extensibility', 'improved workflow',
                        'developer experience', 'lower cost', 'increased adoption', 'competitive advantage',
                        'business value', 'delight users', 'reduce churn', 'boost retention', 'higher engagement',
                        'organic growth', 'improved outcomes', 'strategic priority'],
    'theoretical question': ['theoretical question', 'conceptual query', 'conceptual discussion',
                             'theoretical exploration', 'abstract inquiry', 'philosophical question',
                             'fundamental query', 'conceptual clarification', 'principles question',
                             'theoretical framework', 'concept examination', 'foundational query', 'academic question',
                             'speculative inquiry', 'concept analysis', 'theoretical examination', 'modeling question',
                             'why', 'how does', 'what is', 'explain', 'meaning of', 'purpose of', 'reason for',
                             'basics of', 'understand', 'clarify', 'difference between', 'simple explanation',
                             'overview of', 'in simple terms', 'basic concept', 'logical inquiry', 'strategic question',
                             'analytical query', 'methodological question', 'philosophical query',
                             'intellectual challenge', 'theoretical challenge', 'ontological discussion',
                             'epistemic question', 'metaphysical analysis', 'theoretical discourse',
                             'hypothetical reasoning', 'thought experiment', 'conceptual analysis',
                             'theoretical construct', 'rationalist perspective', 'empiricist critique',
                             'exploratory dialogue', 'assumption examination', 'logical positivism',
                             'critical rationalism', 'paradigm analysis', 'anthropic principle',
                             'rationalism vs empiricism'],
    'security': ['security', 'vulnerability', 'threat', 'exploit', 'risk', 'penetration', 'data breach', 'compromise',
                 'hacking', 'data protection', 'cybersecurity', 'information security', 'privacy concern',
                 'security breach', 'intrusion', 'phishing', 'authentication issue', 'encryption concern', 'safe',
                 'protect', 'secure', 'risk', 'hack', 'privacy', 'leak', 'safe to use', 'data safety', 'secure enough',
                 'password', 'firewall', 'safety concern', 'confidential', 'access control', 'compliance',
                 'cyber attack', 'identity theft', 'security policy', 'malware', 'spyware', 'security audit', 'virus',
                 'trojan', 'worm', 'ransomware', 'keylogger', 'botnet', 'rootkit', 'cryptojacking', 'brute force',
                 'sql injection', 'ddos attack', 'data encryption', 'firewall', 'vpn', 'ssh keys', 'https', 'tls',
                 'saml', 'rbac'],
    'other': []
}


# Preprocessing and cleaning the text
def text_cleaning(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)


# Function to categorize prompts by keywords
def prompt_categorization_by_labelling_keys(cleaned_prompt, labelling_keys):
    for label, keywords in labelling_keys.items():
        for keyword in keywords:
            if keyword in cleaned_prompt.lower():
                return label
    return 'other'  # Default label if no keywords match


# Function to load and preprocess data with categorized prompts
def preprocessing_and_loading_data(file_names):
    all_texts = []
    all_labels = []
    prompts_by_category = {label: [] for label in labelling_keys.keys()}

    for file_name in file_names:
        with open(file_name, 'r') as file:
            data = json.load(file)
            for data1 in data['Sources']:
                for record in data1.get('ChatgptSharing', []):
                    for conversation in record.get('Conversations', []):
                        prompt = conversation.get('Prompt', '')
                        cleaned_prompt = text_cleaning(prompt)
                        all_texts.append(cleaned_prompt)
                        label = prompt_categorization_by_labelling_keys(cleaned_prompt, labelling_keys)
                        all_labels.append(label)
                        prompts_by_category[label].append(prompt)  # Store prompts by category

    for label, prompts in prompts_by_category.items():
        print(f"Category: {label}")
        print(f"Total Prompts: {len(prompts)}")
        print("-------------------------------")

    return all_texts, all_labels, prompts_by_category


def category_prediction(new_prompt, vectorizer, clf, label_encoder):
    cleaned_prompt = text_cleaning(new_prompt)
    vectorized_prompt = vectorizer.transform([cleaned_prompt])
    predicted_label = clf.predict(vectorized_prompt)
    predicted_category = label_encoder.inverse_transform(predicted_label)
    return predicted_category[0]


# Define your file names
file_names = ['discuss.json', 'issues.json', 'commit.json', 'pr_sharings.json', 'hn_sharings.json', 'file_sharing.json']

# Load data
texts, labels, prompts_by_category = preprocessing_and_loading_data(file_names)

# Using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert texts to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)
# Create and train the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
# Evaluate the model
y_pred = clf.predict(X_test)

# Plotting a bar plot for category distribution
categories = list(prompts_by_category.keys())
counts = [len(prompts) for prompts in prompts_by_category.values()]

plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color='skyblue')
plt.ylabel('Counts')
plt.xlabel('Categories')
plt.title('Category Distribution of Prompts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

new_prompt = input("Enter a prompt to classify: ")
predicted_category = category_prediction(new_prompt, vectorizer, clf, label_encoder)
print(f"The predicted category for the prompt is: {predicted_category}")