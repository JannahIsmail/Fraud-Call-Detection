import nltk
import streamlit as st
import toml
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, bigrams
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import string
import re

# Specify the file path
file_path = r'C:\Users\win10user\Documents\fraud call data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Labelling the columns
X = df['Contents']
y = df['Type']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Split the original data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20, random_state=0)

# Apply SMOTE to the training data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler(with_mean=False)  # Disable centering for sparse matrices
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Create a DataFrame from the resampled data
resampled_df = pd.DataFrame(data={'Contents': X_train_resampled, 'Type': y_train_resampled})

# Save the resampled data as a CSV file
resampled_file_path = r'C:\Users\win10user\Downloads\smotedataresult.csv'
resampled_df.to_csv(resampled_file_path, index=False)

# Create a CountVectorizer with n-grams
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
X_ngrams = ngram_vectorizer.fit_transform(df[df['Type'] == 'fraud']['Contents'])

# Convert the sparse matrix to a DataFrame with column names
ngram_df = pd.DataFrame(X_ngrams.toarray(), columns=ngram_vectorizer.get_feature_names_out())

# Calculate the top 10 n-grams for fraud messages
top_ngrams = ngram_df.sum().sort_values(ascending=False).head(10)

# Concatenate all fraud messages into a single string
fraud_messages = df[df['Type'] == 'fraud']['Contents']
fraud_text = ' '.join(fraud_messages)

# Concatenate all normal messages into a single string
normal_messages = df[df['Type'] == 'normal']['Contents']
normal_text = ' '.join(normal_messages)

# Tokenize the text
tokens = word_tokenize(fraud_text)

# Remove stop words and punctuation
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if (word.lower() not in stop_words) and (word.lower() not in string.punctuation)]

# Create bigrams
fraud_bigrams = list(bigrams(filtered_tokens))

# Calculate the top 10 bigrams for fraud messages
top_bigrams = FreqDist(fraud_bigrams).most_common(10)

# Create a frequency dictionary for bigrams
bigram_freq = {}
for bigram in fraud_bigrams:
    bigram_str = ' '.join(bigram)
    bigram_freq[bigram_str] = bigram_freq.get(bigram_str, 0) + 1

import streamlit as st

model = RandomForestClassifier(max_depth=None, min_samples_split=10, n_estimators=50, random_state=0)
model.fit(X_train_resampled_scaled, y_train_resampled)

# TF-IDF vectorization (reuse the existing vectorizer)
def vectorize_input_text(input_text):
    # Tokenize the text
    tokens = word_tokenize(input_text)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if (word.lower() not in stop_words) and (word.lower() not in string.punctuation)]

    # Join the filtered tokens into a string
    preprocessed_text = ' '.join(filtered_tokens)

    # Vectorize the preprocessed text using the existing vectorizer
    input_vectorized = vectorizer.transform([preprocessed_text])

    return input_vectorized

def generate_fraud_calls_visualizations():
    with st.expander("Top 10 Words in Fraud Calls"):
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors1 = plt.cm.PiYG(np.linspace(0, 1, len(top_ngrams)))
        ax1.barh(top_ngrams.index, top_ngrams.values, color=colors1, alpha=0.7, height=0.8)
        ax1.set_title('Top 10 Words in Fraud Calls')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('N-grams')
        fig1.patch.set_alpha(0.0)  
        st.pyplot(fig1)
    
    with st.expander("Top 10 Phrases in Fraud Calls"):
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors2 = plt.cm.PiYG(np.linspace(0, 1, len(top_bigrams)))
        ax2.barh([str(bigram) for bigram, _ in top_bigrams], [count for _, count in top_bigrams], color=colors2, alpha=0.7, height=0.8)
        ax2.set_title('Top 10 Phrases in Fraud Calls')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Bigrams')
        fig2.patch.set_alpha(0.0)  
        st.pyplot(fig2)

    with st.expander("WordCloud for Fraud Messages"):
        wc_fig, wc_ax = plt.subplots(figsize=(8, 4))
        wc = WordCloud(width=800, height=400, background_color=None, mode='RGBA', colormap='PiYG', random_state=42).generate(fraud_text)
        wc_ax.imshow(wc, interpolation='bilinear')
        wc_ax.axis("off")
        wc_fig.set_facecolor('none') 
        st.pyplot(wc_fig)

    with st.expander("WordCloud for Bigrams"):
        wc_bigrams_fig, wc_bigrams_ax = plt.subplots(figsize=(8, 4))
        wc_bigrams = WordCloud(width=800, height=400, background_color=None, mode='RGBA', colormap='PiYG', random_state=42).generate_from_frequencies(bigram_freq)
        wc_bigrams_ax.imshow(wc_bigrams, interpolation='bilinear')
        wc_bigrams_ax.axis("off")
        wc_bigrams_fig.set_facecolor('none') 
        st.pyplot(wc_bigrams_fig)

pages = ["Home", "Top Words from Fraud Calls", "Documentation", "About"]
selected_page = st.sidebar.selectbox("Select Page", pages)

if selected_page == "Home":
    col1, col2 = st.columns([1, 4])

    image_path = "https://64.media.tumblr.com/d597959c5280ed1ba0613674dcd501aa/8be92ff84e25a46d-f2/s2048x3072/22558d2ab28a7f430f86a866a1db24665b278268.pnj" 
    col1.image(image_path, use_column_width=False, width=100)  # Adjust width as needed

    col2.markdown("# Fraud Call Detection App")

    user_input = st.text_input("Enter a phrase or keywords from the phone call:")

    if st.button("Detect"):
        if user_input:
            input_vectorized = vectorize_input_text(user_input)

            if input_vectorized.nnz == 0:
                 st.warning("The input does not contain relevant information. Please enter meaningful text.")
            else:
                 prediction = model.predict(input_vectorized)

                 st.success(f"The input is classified as: {prediction}")
        else:
            st.warning("Please enter a phrase or keywords before detecting.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    gif_url = "https://64.media.tumblr.com/4d60f9517a188d3429af8f449bccfd5b/66f4740aea66cf6e-06/s640x960/f676448fdc127616fd24c9f329bc29bf69859708.gifv"  # Replace with the URL of your GIF
    st.image(gif_url, use_column_width=True)

elif selected_page == "Top Words from Fraud Calls":
    col1, col2 = st.columns([1, 4])
    image_path = "https://64.media.tumblr.com/d597959c5280ed1ba0613674dcd501aa/8be92ff84e25a46d-f2/s2048x3072/22558d2ab28a7f430f86a866a1db24665b278268.pnj" 
    col1.image(image_path, use_column_width=False, width=100)  

    col2.markdown("# Top Words from Fraud Calls")

    generate_fraud_calls_visualizations()

elif selected_page == "Documentation":
    col1, col2 = st.columns([1, 4])
    image_path = "https://64.media.tumblr.com/d597959c5280ed1ba0613674dcd501aa/8be92ff84e25a46d-f2/s2048x3072/22558d2ab28a7f430f86a866a1db24665b278268.pnj" 
    col1.image(image_path, use_column_width=False, width=100) 

    col2.markdown("# Documentation")

    st.markdown("## Home Page")
    st.markdown("The Home page allows users to interact with the Fraud Call Detection App. Users can enter a phrase or keywords from a phone call in the text input, click the 'Detect' button, and receive a classification prediction")
    
    st.markdown("## Top Words from Fraud Calls")
    st.markdown("This page displays visualizations related to the top words and phrases in fraud calls. Users can take note of the words and phrases that are displayed to learn more about the contents of fraud calls.")
    
    st.markdown("## About")
    st.markdown("The About page provides information about the Fraud Call Detection App. It includes the details about the development, the purpose of the app, and any other relevant information.")

    gif_url = "https://64.media.tumblr.com/4d60f9517a188d3429af8f449bccfd5b/66f4740aea66cf6e-06/s640x960/f676448fdc127616fd24c9f329bc29bf69859708.gifv"  # Replace with the URL of your GIF
    st.image(gif_url, use_column_width=True)
    
elif selected_page == "About":
    col1, col2 = st.columns([1, 4])
    image_path = "https://64.media.tumblr.com/d597959c5280ed1ba0613674dcd501aa/8be92ff84e25a46d-f2/s2048x3072/22558d2ab28a7f430f86a866a1db24665b278268.pnj" 
    col1.image(image_path, use_column_width=False, width=100) 

    col2.markdown("# About")

    st.markdown("## Purpose of the App")
    st.markdown("Fraud calls are a serious issue that can lead to financial loss, identity theft, and other negative consequences. The Fraud Call Detection App is designed to help users identify potential fraud calls by analyzing the content of phone calls. The app uses machine learning to classify calls as either normal or fraudulent, providing users with an extra layer of protection against scams and fraudulent activities.")

    st.markdown("## Model Information")
    st.markdown("The Fraud Call Detection App utilizes the Random Forest Classifier algorithm, a machine learning model known for its versatility and high accuracy. The model has been trained on a dataset of phone call contents and can predict whether a given call is likely to be fraudulent or not with an accuracy of 98%.")

    st.markdown("## Learn More :)")
    st.markdown("Watch this informative video to learn more about fraud calls and how to protect yourself:")
    st.video("https://www.youtube.com/watch?v=E4CDuJof5e0")