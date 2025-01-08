import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, message=".*setting.*")

# Data Loading Function
def load_data(pet_file, screening_file, recommendation_file):
    """
    Load the datasets from the given file paths.

    Args:
    - pet_file (str): File path for the pet dataset.
    - screening_file (str): File path for the screening model dataset.
    - recommendation_file (str): File path for the recommendation model dataset.

    Returns:
    - df1 (DataFrame): Loaded pet dataset.
    - df2 (DataFrame): Loaded screening model dataset.
    - df3 (DataFrame): Loaded recommendation model dataset.
    """
    df1 = pd.read_csv(pet_file)
    df2 = pd.read_csv(screening_file)
    df3 = pd.read_csv(recommendation_file)

    # Rename the existing index to 'pet_id' without creating a new column
    df1.index.rename('pet_id', inplace=True)

    return df1, df2, df3

# Preprocessing for Screening Model (df2)
def preprocess_screening_data(df2):
    #  Get the column names (questions)
    questions = pd.Series(df2.columns)

    #  Define question types
    quest_type = {
        'close': [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 33, 35, 37, 39, 41, 43]
    }
    quest_type['open'] = [i for i in range(43) if i not in quest_type['close']]

    task_type = {
        'application': [i for i in range(44)]
    }

    # Helper function to get task type
    def get_task_type(index):
        for task, indices in task_type.items():
            if index in indices:
                return task
        return 'unknown'

    # Helper function to get question type
    def get_quest_type(index):
        for quest, indices in quest_type.items():
            if index in indices:
                return quest
        return 'unknown'

    # Rename columns based on task and question type
    new_column_names = []
    for i, col in enumerate(questions):
        task = get_task_type(i)
        quest = get_quest_type(i)
        new_name = f"{task}_{quest}_{i}"
        new_column_names.append(new_name)
    
    df2.columns = new_column_names

    # Define function to get the original question from the new name
    def get_question(question: str):
        idx = question.split('_')[-1]
        return questions[eval(idx)]

    # Identify columns to drop based on "open" and not equal to "application_open_31"
    columns_to_drop = [col for col in df2.columns if "open" in col and col != "application_open_31"]

    # specific "application_close" columns to drop
    columns_to_drop += ["application_close_33", "application_close_35", "application_close_37", "application_close_39", "application_close_41"]

    # Drop the identified columns
    df2 = df2.drop(columns=columns_to_drop)

    #  Standardize variations of 'None' across the data using a vectorized approach
    none_variations = ['none', 'NONE', 'n/a', 'N/A', '']
    df2 = df2.replace(none_variations + [np.nan], 'None')

    #  Handle previous pet-related columns where 'Previous Pet Species' is 'None'
    previous_pet_columns = [
        'application_close_21', 'application_close_23', 'application_close_25',
        'application_close_27', 'application_close_29', 'application_open_31'
    ]
    df2.loc[df2['application_close_19'] == 'None', previous_pet_columns] = 'None'

    # : Fill 'None' for specific columns with null values
    fill_none = ['application_close_27', 'application_close_29']
    df2[fill_none] = df2[fill_none].fillna('None')

    # Fill 'No Response' for another set of columns with null values
    fill_no = ['application_close_23', 'application_close_25']
    df2[fill_no] = df2[fill_no].fillna('No Response')

    return df2

# Preprocessing for Recommendation Model (df3)
def preprocess_recommendation_data(df3):
    questions = pd.Series(df3.columns)

    quest_type = {'close': [0, 1, 2, 3, 4, 5]}
    quest_type['open'] = [i for i in range(7) if i not in quest_type['close']]
    
    task_type = {'recommendation': [i for i in range(7)]}

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
        new_name = f"{task}_{quest}_{i}"
        new_column_names.append(new_name)
    df3.columns = new_column_names

    def get_question(question: str):
        idx = question.split('_')[-1]
        return questions[eval(idx)]

    return df3
    
def prepare_features(data, categorical_columns, textual_columns, exclude_columns=None):
    """
    Prepares the features by encoding categorical columns and transforming textual columns using TF-IDF.

    Parameters:
        data (pd.DataFrame): The input DataFrame to process.
        categorical_columns (list): List of categorical column names to encode.
        textual_columns (list): List of textual column names to process with TF-IDF.
        exclude_columns (list, optional): List of columns to exclude from processing.

    Returns:
        pd.DataFrame: Transformed DataFrame with original columns removed and processed features included.
    """
    import pandas as pd
    import nltk
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Drop excluded columns if specified
    if exclude_columns:
        data = data.drop(columns=exclude_columns)

    # Initialize OneHotEncoder and LabelEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    label_encoder = LabelEncoder()

    # Process categorical columns
    encoded_dfs = []
    for col in categorical_columns:
        if data[col].nunique() == 2:  # Binary column
            data[col] = label_encoder.fit_transform(data[col])
        else:  # Non-binary categorical column
            encoded_cats = encoder.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out([col]))
            encoded_dfs.append(encoded_df)
            data = data.drop(columns=[col])

    # Concatenate all encoded DataFrames
    if encoded_dfs:
        encoded_df = pd.concat(encoded_dfs, axis=1).reset_index(drop=True)
        data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

    # Define custom stopwords
    nltk.download('stopwords')
    custom_stopwords = set(nltk.corpus.stopwords.words('english'))
    important_words = {'adopt', 'pet', 'neuter', 'vaccinated', 'care', 'companion', 'dog', 'cat'}
    custom_stopwords.difference_update(important_words)
    custom_stopwords = list(custom_stopwords)

    # Function to preprocess and transform textual columns using TF-IDF
    def split_and_tfidf_column(column, data, tfidf_result_df):
        try:
            # Preprocess text (case normalization, deduplication)
            data[column] = data[column].str.lower().apply(lambda x: ' '.join(set(x.split())))
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words=custom_stopwords)
            X = vectorizer.fit_transform(data[column].astype(str))
            feature_names = [f"{column}_{feature}" for feature in vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
            return pd.concat([tfidf_result_df, tfidf_df], axis=1)
        except ValueError as e:
            return tfidf_result_df

    # Initialize a DataFrame to store TF-IDF features
    tfidf_result_df = pd.DataFrame()

    # Process textual columns
    for column in textual_columns:
        tfidf_result_df = split_and_tfidf_column(column, data, tfidf_result_df)

    # Drop original textual columns and concatenate TF-IDF features
    data = data.drop(columns=textual_columns).reset_index(drop=True)
    data = pd.concat([data, tfidf_result_df], axis=1)

    return data
    
def run_process(df1, df2, df3):
    """
    Runs the pipeline for pet, user profile, previous pet, and desired pet data.
    """
    # Process the pet data
    categorical_columns_pet = ['Type', 'Age', 'Gender', 'Vaccinated', 'Spayed/Neutered', 'AdoptionStory']
    textual_columns_pet = ['Extracted_Temperaments']
    exclude_columns_pet = []
    pet_data = prepare_features(df1, categorical_columns_pet, textual_columns_pet, exclude_columns=exclude_columns_pet)

    # Define columns for personal details
    categorical_columns_user = [
        'application_close_0', 'application_close_1', 'application_close_2',
        'application_close_3', 'application_close_5', 'application_close_7',
        'application_close_9', 'application_close_11', 'application_close_13',
        'application_close_15', 'application_close_17'
    ]
    textual_columns_user = []
    exclude_columns_user = [
        'application_close_19', 'application_close_21', 'application_close_23',
        'application_close_25', 'application_close_27', 'application_close_29',
        'application_open_31'
    ]
    user_profile = prepare_features(df2, categorical_columns_user, textual_columns_user, exclude_columns=exclude_columns_user)

    # Define columns for previous pet
    categorical_columns_prev = [
        'application_close_19', 'application_close_21', 'application_close_23',
        'application_close_25', 'application_close_27', 'application_close_29'
    ]
    textual_columns_prev = ['application_open_31']
    exclude_columns_prev = [
        'application_close_0', 'application_close_1', 'application_close_2',
        'application_close_3', 'application_close_5', 'application_close_7',
        'application_close_9', 'application_close_11', 'application_close_13',
        'application_close_15', 'application_close_17'
    ]
    previous_pet = prepare_features(df2, categorical_columns_prev, textual_columns_prev, exclude_columns=exclude_columns_prev)

    # List of categorical columns relevant to the desired features for adopters without experience
    categorical_columns_desired = [
        'recommendation_close_0', 'recommendation_close_1', 'recommendation_close_2',
        'recommendation_close_3', 'recommendation_close_4', 'recommendation_close_5'
    ]
    textual_columns_desired = ['recommendation_open_6']
    exclude_columns_desired = []
    desired_pet = prepare_features(df3, categorical_columns_desired, textual_columns_desired, exclude_columns=exclude_columns_desired)
      # Return the processed data
    return pet_data, user_profile, previous_pet, desired_pet 


def add_adopter_id(df):
    """Add 'adopter_id' to the DataFrame if not already present."""
    if 'adopter_id' not in df.columns:
        df['adopter_id'] = range(1, len(df) + 1)
    return df

def assign_adopter_category(previous_pet):
    """
    Assign 'adopter_category' based on 'application_close_19_None' column.
    
    Args:
    - previous_pet (pd.DataFrame): The previous pet DataFrame.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'adopter_category'.
    """
    if 'application_close_19_None' in previous_pet.columns:
        def determine_adopter_category(row):
            return '0' if row['application_close_19_None'] == 1.0 else '1'
        previous_pet['adopter_category'] = previous_pet.apply(determine_adopter_category, axis=1)
    else:
        raise ValueError("'application_close_19_None' column is missing in previous_pet DataFrame.")
    
    return previous_pet
def merge_adopter_category(user_profile, desired_pet, previous_pet):
    """
    Merge 'adopter_category' from previous_pet into user_profile and desired_pet.
    
    Args:
    - user_profile (pd.DataFrame): The user profile DataFrame.
    - desired_pet (pd.DataFrame): The desired pet DataFrame.
    - previous_pet (pd.DataFrame): The previous pet DataFrame with 'adopter_category'.
    
    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: Updated user_profile and desired_pet DataFrames.
    """
    # Check if 'adopter_category' exists and drop it if necessary to avoid duplication
    if 'adopter_category' in user_profile.columns:
        user_profile.drop('adopter_category', axis=1, inplace=True)
    
    if 'adopter_category' in desired_pet.columns:
        desired_pet.drop('adopter_category', axis=1, inplace=True)

    # Extract only 'adopter_id' and 'adopter_category' columns, removing duplicates
    previous_pet_unique = previous_pet[['adopter_id', 'adopter_category']].drop_duplicates()

    # Merge with user_profile and desired_pet
    user_profile = pd.merge(user_profile, previous_pet_unique, how='left', on='adopter_id', suffixes=('', '_prev'))
    desired_pet = pd.merge(desired_pet, previous_pet_unique, how='left', on='adopter_id', suffixes=('', '_prev'))
    
    return user_profile, desired_pet

def convert_to_dataframe(data, num_columns=None):
    """
    Converts ndarray to a DataFrame, adding default column names if necessary.
    """
    if isinstance(data, np.ndarray):
        if num_columns is None:
            num_columns = data.shape[1]
        columns = [f'col_{i}' for i in range(num_columns)]
        return pd.DataFrame(data, columns=columns)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise ValueError("Input must be either a pandas DataFrame or a numpy ndarray")

def align_dataframe(df, all_columns_list, reference_df=None):
    """
    Align the DataFrame columns with a reference list and fill missing columns with zeros.
    """
    df = convert_to_dataframe(df, num_columns=len(all_columns_list))
    
    # Add missing columns with zeros
    missing_columns = list(set(all_columns_list) - set(df.columns))
    if missing_columns:
        missing_df = pd.DataFrame(0, index=df.index, columns=missing_columns)
        df = pd.concat([df, missing_df], axis=1)
    
    # Reorder columns to match the reference (sorted all_columns_list)
    df = df[all_columns_list]
    
    # Align rows with the reference DataFrame
    if reference_df is not None:
        df = df.reindex(reference_df.index, fill_value=0)
    else:
        df = df.reindex(sorted(df.index), fill_value=0)
    
    return df
    
def run_pipeline(pet_data, user_profile, previous_pet, desired_pet, df2, df3):
    """
    Runs the entire pipeline to prepare the data.
    
    Args:
    - pet_data (pd.DataFrame): The pet data DataFrame.
    - user_profile (pd.DataFrame): The user profile DataFrame.
    - previous_pet (pd.DataFrame): The previous pet DataFrame.
    - desired_pet (pd.DataFrame): The desired pet DataFrame.
    - df2 (pd.DataFrame): The second DataFrame that will include 'adopter_id'.
    - df3 (pd.DataFrame): The third DataFrame that will include 'adopter_id'.
    
    Returns:
    - pd.DataFrame: Processed DataFrames (pet_data, user_profile, previous_pet, desired_pet, df2, df3).
    """
    # Add 'adopter_id' if not present
    user_profile = add_adopter_id(user_profile)
    previous_pet = add_adopter_id(previous_pet)
    desired_pet = add_adopter_id(desired_pet)

    #  Assign 'adopter_category' to previous_pet
    previous_pet = assign_adopter_category(previous_pet)

    #  Merge 'adopter_category' into user_profile and desired_pet
    user_profile, desired_pet = merge_adopter_category(user_profile, desired_pet, previous_pet)

    #  Ensure all dataframes have the same columns
    all_columns = set(pet_data.columns) | set(user_profile.columns) | set(previous_pet.columns) | set(desired_pet.columns)
    all_columns_list = sorted(list(all_columns))

    # Align all DataFrames using the same set of columns
    pet_data = align_dataframe(pet_data, all_columns_list, reference_df=pet_data)
    user_profile = align_dataframe(user_profile, all_columns_list, reference_df=pet_data)
    previous_pet = align_dataframe(previous_pet, all_columns_list, reference_df=pet_data)
    desired_pet = align_dataframe(desired_pet, all_columns_list, reference_df=pet_data)

    #  Add 'adopter_id' to df2 and df3
    df2['adopter_id'] = user_profile['adopter_id']
    df3['adopter_id'] = user_profile['adopter_id']

    return pet_data, user_profile, previous_pet, desired_pet, df2, df3


def compute_similarity(data1, data2):
    """
    Computes the cosine similarity between two datasets.
    """
    return cosine_similarity(data1, data2)

def recommend_pets_with_features(adopter_id, top_n=5, threshold=0.4, recommended_pet_ids=None, user_profile=None, df1=None, df2=None, df3=None, previous_pet=None, desired_pet=None, pet_data=None):
    if recommended_pet_ids is None:
        recommended_pet_ids = set()
    
    if user_profile is None:
        raise ValueError("user_profile DataFrame must be provided.")
    
    adopter_row = user_profile[user_profile['adopter_id'] == adopter_id]
    if adopter_row.empty:
        print(f"Adopter ID {adopter_id} not found in the database.")
        return []
    
    adopter_category = adopter_row['adopter_category'].values[0]
    category_mapping = {0: 'New', 1: 'Experienced'}
    adopter_category_name = category_mapping.get(int(adopter_category), 'Unknown')
    print(f"Adopter ID {adopter_id} falls under the category: {adopter_category_name}")

    if 'application_close_0' in df2.columns:
        adopter_profile = df2[df2['adopter_id'] == adopter_id][[
            'application_close_0', 'application_close_1', 'application_close_2',
            'application_close_3', 'application_close_5', 'application_close_7',
            'application_close_9', 'application_close_11', 'application_close_13',
            'application_close_15', 'application_close_17'
        ]]
    else:
        print("Error: Required columns missing in df2.")
    
    if 'application_close_19' in df2.columns:
        previous_pets = df2[df2['adopter_id'] == adopter_id][[
            'application_close_19', 'application_close_21', 'application_close_23',
            'application_close_25', 'application_close_27', 'application_close_29',
            'application_open_31'
        ]]
    else:
        print("Error: Required columns missing in df2.")
    
    if 'recommendation_close_0' in df3.columns:
        adopter_preferences = df3[df3['adopter_id'] == adopter_id][[
            'recommendation_close_0', 'recommendation_close_1',
            'recommendation_close_2', 'recommendation_close_3',
            'recommendation_close_4', 'recommendation_close_5',
            'recommendation_open_6'
        ]]
        
        type_mapping = {0: 'Cat', 1: 'Dog'}
        adopter_preferences['recommendation_close_0'] = adopter_preferences['recommendation_close_0'].map(type_mapping).fillna('Unknown')

    else:
        print("Error: Required columns missing in df3.")

    if adopter_category == "1":
        return _recommend_for_experienced(adopter_row, top_n, threshold, recommended_pet_ids, adopter_preferences, previous_pet, desired_pet, pet_data, df1, user_profile, type_mapping)

    elif adopter_category == "0":
        return _recommend_for_new(adopter_row, top_n, threshold, recommended_pet_ids, adopter_preferences, previous_pet, desired_pet, pet_data, df1, user_profile, type_mapping)

    print("Adopter category is unknown.")
    return []

def _recommend_for_experienced(adopter_row, top_n, threshold, recommended_pet_ids, adopter_preferences, previous_pet, desired_pet, pet_data, df1, user_profile, type_mapping):
    current_adopter_previous_pet = previous_pet[previous_pet['adopter_id'] == adopter_row['adopter_id'].values[0]]
    other_previous_pet = previous_pet[previous_pet['adopter_id'] != adopter_row['adopter_id'].values[0]]
    desired_pet_data = desired_pet[desired_pet['adopter_id'] == adopter_row['adopter_id'].values[0]]

    if current_adopter_previous_pet.empty or desired_pet_data.empty:
        print("Error: Either previous pets or desired pets data is empty.")
        return []

    item_similarity = compute_similarity(current_adopter_previous_pet.iloc[:, 1:], desired_pet_data.iloc[:, 1:])
    print(f"Previous-Preference Similarity Scores: {item_similarity.flatten()}")
    if np.any(item_similarity >= threshold):
        similarity_scores = compute_similarity(current_adopter_previous_pet.iloc[:, 1:], pet_data.iloc[:, 1:])
        recommended_pets = get_top_n_pets_with_fallback(similarity_scores, top_n, threshold, recommended_pet_ids, df1, user_profile, adopter_preferences, type_mapping)
        return recommended_pets
    else:
        print(f"Threshold of {threshold} not met. Using User-Based Filtering.")
        user_similarity = compute_similarity(desired_pet_data.iloc[:, 1:], other_previous_pet.iloc[:, 1:])
        print(f"User-Based Similarity Scores: {user_similarity.flatten()}")
        recommended_pets = get_top_n_pets_with_fallback(user_similarity, top_n, threshold, recommended_pet_ids, df1, adopter_preferences, user_profile, type_mapping)
        return recommended_pets

def _recommend_for_new(adopter_row, top_n, threshold, recommended_pet_ids, adopter_preferences, previous_pet, desired_pet, pet_data, df1, user_profile, type_mapping):
    user_based_similarity = compute_similarity(
        user_profile[user_profile['adopter_category'] == '0'].iloc[:, 1:],
        user_profile[user_profile['adopter_category'] == '1'].iloc[:, 1:]
    )
    item_based_similarity = compute_similarity(
        desired_pet[desired_pet['adopter_category'] == '0'].iloc[:, 1:], pet_data.iloc[:, 1:]
    )

    print(f"User-Based Similarity Score: {np.max(user_based_similarity)}")
    print(f"Item-Based Similarity Score: {np.max(item_based_similarity)}")

    if np.max(user_based_similarity) >= np.max(item_based_similarity):
        print("User-Based Similarity is higher.")
        similarity_scores = compute_similarity(
            previous_pet[previous_pet['adopter_category'] == '1'].iloc[:, 1:], pet_data.iloc[:, 1:]
        )
        recommended_pets = get_top_n_pets_with_fallback(similarity_scores, top_n, threshold, recommended_pet_ids, df1, user_profile, adopter_preferences, type_mapping)
        return recommended_pets
    else:
        print("Item-Based Similarity is higher.")
        recommended_pets = get_top_n_pets_with_fallback(item_based_similarity, top_n, threshold, recommended_pet_ids, df1, user_profile, adopter_preferences, type_mapping)
        return recommended_pets

def get_top_n_pets_with_fallback(similarity_scores, top_n, threshold, recommended_pet_ids, df1, user_profile, adopter_preferences, type_mapping):

    """
    Get the top N pets based on similarity scores, ensuring no empty recommendations.
    If no pets meet the threshold, return the closest ones.
    """
    recommended_pets = []

    # Ensure similarity_scores is a 1D array (flattened)
    similarity_scores = np.array(similarity_scores).flatten()

    # Sort the indices based on similarity scores in descending order
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Get adopter preferences for pet type
    desired_pet_type = adopter_preferences.iloc[0]['recommendation_close_0']  # e.g., 'Dog'

    # Filter pets based on the adopter's desired pet type
    filtered_pets = df1[df1['Type'] == desired_pet_type]  # Filter pets of the correct type

    if filtered_pets.empty:  # If no pets match the desired type, expand the search to all pets
        print(f"No pets found matching desired pet type: {desired_pet_type}. Recommending all pets.")
        filtered_pets = df1

    # Ensure the sorted indices are within the bounds of filtered_pets
    sorted_indices = sorted_indices[sorted_indices < len(filtered_pets)]

    # Recommend pets above the threshold
    for pet_idx in sorted_indices[:top_n]:
        if pet_idx >= len(filtered_pets):
            continue  

        pet_id = filtered_pets.index[pet_idx]  
        if similarity_scores[pet_idx] >= threshold and pet_id not in recommended_pet_ids:
            recommended_pet_ids.add(pet_id)
            recommended_pets.append(pet_id)
          
            pet_type = type_mapping.get(filtered_pets.loc[pet_id, 'Type'], 'Unknown')
        
            print(f"\nRecommended Pet ID: {pet_id}")
            print(f"Pet Features:")
            print(f"Type: {pet_type}")
            print(f"Age: {filtered_pets.loc[pet_id, 'Age']}")
            print(f"Gender: {filtered_pets.loc[pet_id, 'Gender']}")
            print(f"Vaccinated: {filtered_pets.loc[pet_id, 'Vaccinated']}")
            print(f"Spayed/Neutered: {filtered_pets.loc[pet_id, 'Spayed/Neutered']}")
            print(f"Adoption Story: {filtered_pets.loc[pet_id, 'AdoptionStory']}")
            print(f"Temperament: {filtered_pets.loc[pet_id, 'Extracted_Temperaments']}\n")

        if len(recommended_pets) >= top_n:
            break

    # Fallback in case no recommended pets met the similarity
    if not recommended_pets:
        print("No pets met the similarity score. Providing fallback recommendations.")
        for pet_idx in sorted_indices[:top_n]:
            if pet_idx >= len(filtered_pets):
                continue  # Skip if the index is out of range

            pet_id = filtered_pets.index[pet_idx]  # Correctly access the pet ID from filtered pets
            recommended_pets.append(pet_id)
            # Map the numeric pet type to a human-readable string
            pet_type = type_mapping.get(filtered_pets.loc[pet_id, 'Type'], 'Unknown')
            # Print pet features neatly in fallback
            print(f"\nFallback Recommended Pet ID: {pet_id}")
            print(f"Pet Features:")
            print(f"Type: {pet_type}")
            print(f"Age: {filtered_pets.loc[pet_id, 'Age']}")
            print(f"Gender: {filtered_pets.loc[pet_id, 'Gender']}")
            print(f"Vaccinated: {filtered_pets.loc[pet_id, 'Vaccinated']}")
            print(f"Spayed/Neutered: {filtered_pets.loc[pet_id, 'Spayed/Neutered']}")
            print(f"Adoption Story: {filtered_pets.loc[pet_id, 'AdoptionStory']}")
            print(f"Temperament: {filtered_pets.loc[pet_id, 'Extracted_Temperaments']}\n")

    return recommended_pets
    
# full  function
def process_and_recommend(
    pet_file: str, screening_file: str, recommendation_file: str,
    adopter_id: int, top_n: int = 2, threshold: float = 0.4
):
    """
    Load data, preprocess, run pipeline, and recommend pets.
    
    Args:
        pet_file (str): Path to the pet data CSV file.
        screening_file (str): Path to the screening model CSV file.
        recommendation_file (str): Path to the recommendation model CSV file.
        adopter_id (int): ID of the adopter for recommendations.
        top_n (int): Number of recommendations to return.
        threshold (float): Threshold for recommendation filtering.
        
    Returns:
        list: List of recommended pets.
    """
    import reco
    
    df1, df2, df3 = reco.load_data(pet_file, screening_file, recommendation_file)
    
    # Preprocess data
    df2 = reco.preprocess_screening_data(df2)
    df3 = reco.preprocess_recommendation_data(df3)
    
    # Process data
    pet_data, user_profile, previous_pet, desired_pet = reco.run_process(df1, df2, df3)
    
    # Run pipeline
    pet_data, user_profile, previous_pet, desired_pet, df2, df3 = reco.run_pipeline(
        pet_data, user_profile, previous_pet, desired_pet, df2, df3
    )
    
    # Generate recommendations
    recommended_pets = reco.recommend_pets_with_features(
        adopter_id, top_n=top_n, threshold=threshold, user_profile=user_profile,
        df1=df1, df2=df2, df3=df3, previous_pet=previous_pet, 
        desired_pet=desired_pet, pet_data=pet_data
    )
    
    return recommended_pets
