import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import nltk

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# 1. Load the data
df = pd.read_excel('Data_Set.xlsx')  

# 2. Clean the text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply cleaning
df['title_clean'] = df['Title'].apply(clean_text)
df['description_clean'] = df['Description'].apply(clean_text)

# 3. Combine title and description (making title 1.5x heavier)
df['combined'] = df['title_clean'] + " " + df['title_clean'] + " " + df['description_clean']

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# 5. Function to get top similar titles based on title input
def get_top_similar_titles_by_title(title_name, top_n=int(input("Enter number of similar titles you want to see: "))):
    try:
        index = df[df['Title'].str.strip().str.casefold() == title_name.strip().casefold()].index[0]
    except IndexError:
        print(f"Title '{title_name}' not found!")
        return []

    cosine_similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()

    rating_to_match = df.loc[index, 'rating']
    rating_matches = (df['rating'] == rating_to_match).astype(float)

    boost_factor = 0.1
    cosine_similarities += rating_matches * boost_factor

    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    similar_scores = cosine_similarities[similar_indices]
    similar_titles = df.iloc[similar_indices]['Title'].values
    description_of_the_similar_titles = df.iloc[similar_indices]['Description'].values
    return list(zip(similar_titles, description_of_the_similar_titles))


# 6. Usage
title_to_search = input("Enter Title name: ").strip().casefold()  # <-- Enter the title you want to search
print(f"Top similar titles to: {title_to_search}")
for title, description in get_top_similar_titles_by_title(title_to_search):
    print(f"Title: {title} | Description: {description}")
