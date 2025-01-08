import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from functools import lru_cache
warnings.filterwarnings("ignore")

class PetRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        
    @lru_cache(maxsize=128)
    def preprocess_text(self, text):
        """Cache text preprocessing results"""
        return ' '.join(set(str(text).lower().split()))
    
    def preprocess_data(self, df, categorical_cols, text_cols):
        """Vectorized preprocessing"""
        # Handle missing values efficiently
        df = df.fillna({'Type': 'Unknown', 'Age': 'Unknown', 'Gender': 'Unknown'})
        
        # Vectorized text preprocessing
        for col in text_cols:
            df[col] = df[col].apply(self.preprocess_text)
        
        # Efficient categorical encoding
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

    def compute_similarity(self, user_features, pet_features):
        """Compute weighted similarity"""
        # Add feature importance weights
        weights = np.array([
            1.5 if 'type' in col else
            1.2 if 'age' in col else
            1.0 for col in self.encoder.get_feature_names_out()
        ])
        
        weighted_user = user_features * weights
        weighted_pet = pet_features * weights
        
        return cosine_similarity(weighted_user, weighted_pet)

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
                    recommendations.append({
                        'pet_id': pet_data.index[idx],
                        'similarity': float(similarities[0][idx]),
                        'details': pet_data.iloc[idx].to_dict()
                    })
                    
            return recommendations if recommendations else self.fallback_recommendations(pet_data, top_n)
            
        except Exception as e:
            print(f"Error in recommendation process: {e}")
            return []

    def fallback_recommendations(self, pet_data, top_n):
        """Provide fallback recommendations based on popularity"""
        return [{
            'pet_id': idx,
            'similarity': 0.0,
            'details': row.to_dict()
        } for idx, row in pet_data.head(top_n).iterrows()]

# Example usage
if __name__ == "__main__":
    # Load sample data
    pet_data = pd.DataFrame({
        'Type': ['Dog', 'Cat', 'Dog'],
        'Age': ['Puppy', 'Adult', 'Adult'],
        'Gender': ['Male', 'Female', 'Male'],
        'Extracted_Temperaments': ['playful, friendly', 'calm, independent', 'active, loyal']
    })
    
    user_profile = pd.DataFrame({
        'Type': ['Dog'],
        'Age': ['Puppy'],
        'Gender': ['No preference'],
        'Temperament': ['playful, friendly']
    })
    
    # Create recommender and get recommendations
    recommender = PetRecommender()
    recommendations = recommender.get_recommendations(user_profile, pet_data)
    
    # Print results
    for rec in recommendations:
        print(f"\nPet ID: {rec['pet_id']}")
        print(f"Similarity Score: {rec['similarity']:.2f}")
        print("Details:", rec['details'])