import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from functools import lru_cache
import json
import sys
import traceback

warnings.filterwarnings("ignore")

class PetRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        
    @lru_cache(maxsize=128)
    def preprocess_text(self, text):
        """Cache text preprocessing results"""
        return ' '.join(set(str(text).lower().split()))
    
    def preprocess_data(self, df, categorical_cols, text_cols):
        """Vectorized preprocessing"""
        try:
            # Handle missing values efficiently
            df = df.fillna({'Type': 'Unknown', 'Age': 'Unknown', 'Gender': 'Unknown'})
            
            # Vectorized text preprocessing
            for col in text_cols:
                df[col] = df[col].apply(self.preprocess_text)
            
            # Efficient categorical encoding
            if len(df) > 0:  # Check if dataframe is not empty
                encoded_features = self.encoder.fit_transform(df[categorical_cols])
                encoded_df = pd.DataFrame(
                    encoded_features,
                    columns=self.encoder.get_feature_names_out(categorical_cols)
                )
                
                # Combine features
                processed_df = pd.concat([
                    encoded_df,
                    pd.DataFrame(
                        self.tfidf.fit_transform(df[text_cols].fillna('').agg(' '.join, axis=1)).toarray(),
                        columns=self.tfidf.get_feature_names_out()
                    )
                ], axis=1)
                
                return self.scaler.fit_transform(processed_df)
            else:
                raise ValueError("Empty dataframe provided for preprocessing")
                
        except Exception as e:
            raise Exception(f"Error in preprocessing data: {str(e)}")

    def compute_similarity(self, user_features, pet_features):
        """Compute weighted similarity"""
        try:
            # Add feature importance weights
            weights = np.array([
                1.5 if 'type' in col else
                1.2 if 'age' in col else
                1.0 for col in self.encoder.get_feature_names_out()
            ])
            
            weighted_user = user_features * weights
            weighted_pet = pet_features * weights
            
            return cosine_similarity(weighted_user, weighted_pet)
        except Exception as e:
            raise Exception(f"Error in computing similarity: {str(e)}")

    def get_recommendations(self, user_profile, pet_data, top_n=5, threshold=0.4):
        """Get pet recommendations with fallback strategy"""
        try:
            # Process user and pet data
            user_features = self.preprocess_data(
                user_profile,
                ['Type', 'Age', 'Gender'],
                ['Temperament']
            )
            
            pet_features = self.preprocess_data(
                pet_data,
                ['Type', 'Age', 'Gender'],
                ['Extracted_Temperaments']
            )
            
            # Compute similarities
            similarities = self.compute_similarity(user_features, pet_features)
            
            # Get recommendations
            top_matches = (-similarities[0]).argsort()[:top_n]
            recommendations = []
            
            for idx in top_matches:
                if similarities[0][idx] >= threshold:
                    pet_details = pet_data.iloc[idx].to_dict()
                    # Ensure all values are JSON serializable
                    pet_details = {k: str(v) if isinstance(v, (np.int64, np.float64)) else v 
                                 for k, v in pet_details.items()}
                    recommendations.append({
                        'pet_id': int(idx),
                        'similarity': float(similarities[0][idx]),
                        'details': pet_details
                    })
                    
            return recommendations if recommendations else self.fallback_recommendations(pet_data, top_n)
            
        except Exception as e:
            raise Exception(f"Error in generating recommendations: {str(e)}")

    def fallback_recommendations(self, pet_data, top_n):
        """Provide fallback recommendations based on popularity"""
        try:
            return [{
                'pet_id': int(idx),
                'similarity': 0.0,
                'details': {k: str(v) if isinstance(v, (np.int64, np.float64)) else v 
                          for k, v in row.to_dict().items()}
            } for idx, row in pet_data.head(top_n).iterrows()]
        except Exception as e:
            raise Exception(f"Error in fallback recommendations: {str(e)}")

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"Empty CSV file: {file_path}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {str(e)}")

def process_and_recommend(recommendation_file, screening_file, pet_types_file, adopter_id):
    try:
        # Load data
        recommendation_df = load_data(recommendation_file)
        screening_df = load_data(screening_file)
        pet_df = load_data(pet_types_file)

        # Create user profile
        user_profile = pd.DataFrame({
            'Type': [recommendation_df['1. What type of pet are you interested in adopting?'].iloc[0]],
            'Age': [recommendation_df['2. What age range of pet are you interested in?'].iloc[0]],
            'Gender': [recommendation_df['3. Do you have a preference for the pet\'s gender?'].iloc[0]],
            'Temperament': [recommendation_df['7. What temperament are you looking for in a pet?'].iloc[0]]
        })

        # Create recommender and get recommendations
        recommender = PetRecommender()
        recommendations = recommender.get_recommendations(user_profile, pet_df)
        
        # Ensure the output is valid JSON
        result = {
            "status": "success",
            "recommendations": recommendations
        }
        
        # Convert to JSON string and print
        print(json.dumps(result))
        return

    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e)
        }
        print(json.dumps(error_result))
        return

if __name__ == "__main__":
    try:
        if len(sys.argv) != 5:
            raise ValueError("Incorrect number of arguments. Usage: python reco.py <recommendation_file> <screening_file> <pet_types_file> <adopter_id>")
        
        recommendation_file = sys.argv[1]
        screening_file = sys.argv[2]
        pet_types_file = sys.argv[3]
        adopter_id = int(sys.argv[4])
        
        process_and_recommend(recommendation_file, screening_file, pet_types_file, adopter_id)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e)
        }
        print(json.dumps(error_result))