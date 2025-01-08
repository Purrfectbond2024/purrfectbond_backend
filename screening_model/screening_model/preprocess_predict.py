import pandas as pd
import numpy as np
import regex as re
import string
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import contractions
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
import json
import joblib

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Get the model directory from environment variable, or use the screening_model directory as fallback
MODEL_DIR = os.getenv('MODEL_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screening_model')

def load_modeling_columns():
    """
    Load the modeling columns from the modeling_columns.json file in the MODEL_DIR directory.
    """
    modeling_columns_path = os.path.join(MODEL_DIR, 'modeling_columns.json')
    with open(modeling_columns_path, 'r') as file:
        modeling_columns = json.load(file)
    return modeling_columns

# Preprocessing Functions
def drop_columns(data):
    return data.drop(['Username','Full Name:'], axis = 1, errors = 'ignore') # change columns to drop depending on the form 

def rename_columns(data):
    questions = pd.Series(data.columns)
    quest_type = {
        'close': [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 33, 35, 37, 39, 41],
    }
    quest_type['open'] = [i for i in range(43) if i not in quest_type['close']]
    task_type = {'application': [i for i in range(43)]}

    def get_task_type(index):
        for task, indices in task_type.items():
            if index in indices:
                return task
        return 'unknown'

    def get_quest_type(index):
        for quest, indices in quest_type.items():
            if index in indices:
                return quest
        return 'unknown'

    new_column_names = []
    for i, col in enumerate(questions):
        task = get_task_type(i)
        quest = get_quest_type(i)
        new_column_names.append(f"{task}_{quest}_{i}")
    data.columns = new_column_names
    return data

def apply_column_updates(data):
    updates = {
    'application_close_7': {'Own': ('application_open_8', 'None')},
    'application_close_3': {'No': ('application_open_4', 'None')},
    'application_close_9': {'Yes': ('application_open_10', 'None')},
    'application_close_11': {'Indoors': ('application_open_12', 'None')},
    'application_close_13': {'No': ('application_open_14', 'None')},
    'application_close_15': {'No': ('application_open_16', 'None')},
    'application_close_37': {'Yes': ('application_open_38', 'None')},
    'application_close_41': {'Yes': ('application_open_42', 'None')},
    'application_close_5': {'No': ('application_open_6', 'Not interested')},
    'application_close_29': {'Not vaccinated': ('application_open_30', 'No Vaccine')},
    'application_close_27': {'Not yet spayed or neutered': ('application_open_28', 'No Response')},
    'application_close_35': {'No': ('application_open_36', 'No Response')}
    }
    for condition_col, condition_dict in updates.items():
        for condition_val, (target_col, target_val) in condition_dict.items():
            if condition_col in data.columns and target_col in data.columns:
                data.loc[data[condition_col] == condition_val, target_col] = target_val
    return data

def handle_null_values(data):
    none_variations = ['none', 'NONE', 'n/a', 'N/A', '']
    data.replace(none_variations + [np.nan], 'None', inplace = True)
    previous_pet_columns = [
        'application_open_20', 'application_close_21', 'application_open_22',
        'application_close_23', "application_open_24", 'application_close_25',
        'application_open_26', 'application_close_27', 'application_open_28',
        'application_close_29', 'application_open_30', 'application_open_31',
        'application_open_32'
    ]
    data.loc[data['application_close_19'] == 'None', previous_pet_columns] = 'None'
    data.loc[
        data['application_close_27'].isin(['Neutered', 'Spayed']) & data['application_open_28'].isnull(),
        'application_open_28'
    ] = 'No Response'
    fill_none = ['application_close_27', 'application_close_29']
    data[fill_none] = data[fill_none].fillna('None')
    fill_no = [
        'application_open_30', 'application_open_36', 'application_open_40',
        'application_open_18', 'application_open_34', 'application_open_42', 'application_open_14',
        'application_open_16', 'application_open_20', 'application_open_22', 'application_close_23',
        'application_open_24', 'application_close_25', 'application_open_38'
    ]
    data[fill_no] = data[fill_no].fillna('No Response')
    return data  

# Text Preprocessing Classes
class TextPreprocessor:
    def __init__(self):
        self.transforms = [
            PreserveImportant(),
            LowerCase(),
            HandleMispellings(),
            HandleContractions(),
            RemovePunctuations(),
            RemoveSpecialCharacters(),
            RemoveEmojis(),
            RemoveStopWords(),
            Lemmatize(),
            RestoreImportant(),
        ]

    def __call__(self, text):
        for transform in self.transforms:
            text = transform(text)
        return text
        
important = {
    'Not interested': 'Not interested', 
    'No Vaccine': 'No Vaccine', 
    'No Response': 'No Response'
}
class PreserveImportant:
    def __call__(self, text):
        for phrase in important.keys():
            placeholder = f'__{phrase.replace(" ", "_")}__'
            text = re.sub(r'\b' + re.escape(phrase) + r'\b', placeholder, text)
        return text
class RestoreImportant:
    def __call__(self, text):
        for phrase in reversed(important.keys()): 
            placeholder = f'__{phrase.replace(" ", "_")}__'
            text = text.replace(placeholder, phrase)
        return text
class LowerCase:
    def __call__(self, text):
        return text.lower()
class RemovePunctuations:
    def __init__(self):
        self.punctuations = string.punctuation
        
    def __call__(self, text):
        return text.translate(str.maketrans('', '', self.punctuations))
class Tokenize:
    def __init__(self, tokenize_fn=None):
        self.tokenize_fn = tokenize_fn

    def __call__(self, text):
        if self.tokenize_fn is None:
            return text.split(' ')
        return self.tokenize_fn(text)
class RemoveStopWords:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def __call__(self, text):
        words = text.split(' ')
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
class Stem:
    def __init__(self, stemmer=PorterStemmer()):
        self.stemmer = stemmer

    def __call__(self, text):
        words = text.split(' ')
        stemmed = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed)
class Lemmatize:
    def __init__(self, lemmatizer=None):
        self.lemmatizer = lemmatizer or WordNetLemmatizer()

    def __call__(self, text):
        if "__" in text:
            return text
        words = word_tokenize(text)
        pos_tagged = pos_tag(words)
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) or word
            for word, pos in pos_tagged
        ]
        return ' '.join(lemmatized)

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

class RemoveSpecialCharacters:
    def __init__(self, pattern=None):
        if pattern is None:
            self.pattern = r'[^a-zA-Z0-9\s]'
        else:
            self.pattern = pattern
            
    def __call__(self, text):
        return re.sub(self.pattern, '', text)
class RemoveEmojis:
    def __call__(self, text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
class HandleMispellings:
    def __call__(self, text):
        words = text.split(' ')
        corrected = [str(TextBlob(word).correct()) for word in words]
        return ' '.join(corrected)
class HandleContractions:
    def __call__(self, text):
        return contractions.fix(text)
class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, text):
        for t in self.transforms:
            text = t(text)
        return text

def preprocess_open_columns(data):
    open_columns = [col for col in data.columns if 'open' in col]
    text_preprocessor = TextPreprocessor()
    for column in open_columns:
        data[column] = data[column].apply(lambda x: text_preprocessor(x) if pd.notnull(x) else x)
    return data

# Load the saved vectorizer with absolute path
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))

# Transform real-time data using the saved vectorizer
def apply_tfidf(data):
    open_columns = [col for col in data.columns if 'open' in col]
    tfidf_dfs = []
    for column in open_columns:
        X = tfidf_vectorizer.transform(data[column].astype(str)) 
        feature_names = [f"{column}_{feature}" for feature in tfidf_vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names, index=data.index)  #
        tfidf_dfs.append(tfidf_df)
    data = data.drop(columns=open_columns)
    data = pd.concat([data] + tfidf_dfs, axis=1)
    return data

def preprocess_close_columns(data):
    close_columns = [col for col in data.columns if 'close' in col]
    age_mapping = {
        '15-25 years old': 1,
        '26-35 years old': 2,
        '36-45 years old': 3,
        '46-55 years old': 4,
        '56-65 years old': 5,
        '66-75 years old': 6
    }
    if 'application_close_0' in close_columns:
        data.loc[:, 'application_close_0'] = data['application_close_0'].map(age_mapping)

    spay_neuter_mapping = {
        'Spayed': 2,
        'Neutered': 2,
        'Not yet spayed or neutered': 1,
        'None': 0
    }
    if 'application_close_27' in close_columns:
        data.loc[:, 'application_close_27'] = data['application_close_27'].apply(
            lambda x: max(spay_neuter_mapping.get(s.strip(), 0) for s in x.split(';'))
        )

    vaccination_mapping = {
        'Fully Vaccinated': 3,
        'Vaccinated': 3,
        'Partially vaccinated': 2,
        'Not vaccinated': 1,
        'None': 0
    }
    if 'application_close_29' in close_columns:
        data.loc[:, 'application_close_29'] = data['application_close_29'].apply(
            lambda x: max(vaccination_mapping.get(s.strip(), 0) for s in x.split(';'))
        )

    one_hot_columns = [
        'application_close_1', 'application_close_2', 'application_close_19',
        'application_close_21', 'application_close_23', 'application_close_25'
    ]
    one_hot_columns = [col for col in one_hot_columns if col in close_columns]
    data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

    binary_columns = [
        'application_close_3', 'application_close_5', 'application_close_9',
        'application_close_13', 'application_close_15', 'application_close_17',
        'application_close_33', 'application_close_35', 'application_close_37',
        'application_close_39', 'application_close_41'
    ]
    binary_columns = [col for col in binary_columns if col in close_columns]
    for column in binary_columns:
        if column in data.columns:
            data.loc[:, column] = data[column].map({'Yes': 1, 'No': 0})

    if 'application_close_7' in close_columns:
        data.loc[:, 'application_close_7'] = data['application_close_7'].map({'Own': 1, 'Rent': 0})
    if 'application_close_11' in close_columns:
        data.loc[:, 'application_close_11'] = data['application_close_11'].map({'Indoors': 1, 'Outdoors': 0})

    return data

# Consolidate Variations for Specific Features
def consolidate_variations(data):
    # Identify variations of `application_close_19`
    variations = [
        "application_close_19_Cat",
        "application_close_19_Dog",
        "application_close_19_Dog;None",
        "application_close_19_Cat;None",
        "application_close_19_None"
    ]
    consolidated_column = "application_close_19_Cat;Dog"

    # Check if any of the variations are present
    if any(variation in data.columns for variation in variations):
        # Consolidate variations by summing the one-hot encoded columns
        data[consolidated_column] = data.get("application_close_19_Cat", 0) + data.get("application_close_19_Dog", 0)
    else:
        # If none of the variations are present, fill with a default value
        data[consolidated_column] = 0
    
    # Drop the individual variations to avoid redundancy
    data.drop(columns=[col for col in variations if col in data.columns], inplace=True, errors='ignore')
    return data

# Feature Adjustments for Real-Time Data
def adjust_features(data, modeling_columns):
    missing_features = [feature for feature in modeling_columns if feature not in data.columns]
    if missing_features:
        missing_df = pd.DataFrame(0, index=data.index, columns=missing_features)
        data = pd.concat([data, missing_df], axis=1)
    return data[modeling_columns]

# Main preprocessing pipeline
def preprocess_data(data):
    data = drop_columns(data)
    data = rename_columns(data)              
    data = apply_column_updates(data)       
    data = handle_null_values(data)          
    data = preprocess_open_columns(data)   
    data = apply_tfidf(data)                 
    data = preprocess_close_columns(data)   
    data = consolidate_variations(data)       
    modeling_columns = load_modeling_columns()  
    data = adjust_features(data, modeling_columns)  
    return data

# Update the predict_outcome function to use MODEL_DIR
def predict_outcome(data, threshold = 0.8):
    model_path = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
    model = joblib.load(model_path)
    probabilities = model.predict_proba(data) 
    predictions = (probabilities[:, 1] >= threshold).astype(int)  
    outcome = ["Approve" if pred == 1 else "Disapprove" for pred in predictions]
    confidence = probabilities[:, 1]  
    return outcome, confidence

# Full Workflow
def full_pipeline(input_data, threshold=0.8): 
    processed_data = preprocess_data(input_data)
    outcomes, confidence = predict_outcome(processed_data, threshold)  
    for i, (outcome, conf) in enumerate(zip(outcomes, confidence), start=1):
        print(f"Record {i}: Outcome = {outcome}, Confidence = {conf:.2f}")
    return outcomes, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <csv_filename>")
        sys.exit(1)
    
    csv_filename = sys.argv[1]
    
    try:
        # Load the CSV file
        input_data = pd.read_csv(csv_filename)
        
        # Process the data using the pipeline
        data = full_pipeline(input_data, threshold=0.8)
        
        print(f"Processed data saved to {data}")
    except Exception as e:
        print(f"Error processing file: {e}")
