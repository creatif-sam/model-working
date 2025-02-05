# -- coding: utf-8 --
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from transformers import MarianMTModel, MarianTokenizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import spacy
from langdetect import detect
from collections import Counter

# Page configuration
st.set_page_config(page_title="Generalized Text Analysis", layout="wide")

# Load models with caching
@st.cache_resource
def load_translation_model():
    return MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en"), \
           MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# File upload section
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload File", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read and preprocess data based on file type
    if uploaded_file.name.endswith("xlsx"):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset to help users identify the data
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Generalized Text Columns Detection
    text_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    if len(text_columns) == 0:
        st.error("No text columns detected in the uploaded file.")
    else:
        # Ask users to select the column to perform analysis on
        selected_column = st.selectbox("Select a column for analysis", text_columns)

        # Data cleaning (basic)
        def clean_entry(entry):
            return str(entry).strip()

        data[selected_column] = data[selected_column].apply(clean_entry)

        # Language detection for percentage of French words
        st.header("Language Analysis")
        total_entries = len(data[selected_column].dropna())
        french_count = sum(1 for text in data[selected_column] if detect(text) == 'fr')
        french_percentage = (french_count / total_entries) * 100 if total_entries > 0 else 0
        
        st.write(f"Percentage of French entries: **{french_percentage:.2f}%**")

        # Translation section (optional)
        st.header("Text Translation")
        if st.checkbox("Show Translation Progress"):
            tokenizer, model = load_translation_model()

            with st.spinner('Translating content...'):
                data[f'{selected_column}_translated'] = data[selected_column].apply(
                    lambda x: tokenizer.decode(
                        model.generate(**tokenizer(x, return_tensors="pt", truncation=True))[0],
                        skip_special_tokens=True
                    ) if pd.notnull(x) else x
                )
            st.success("Translation completed!")

            # Word Frequency Analysis Before & After Translation
            def get_word_frequencies(texts):
                words = " ".join(texts.dropna()).split()
                return Counter(words)
            
            original_word_freq = get_word_frequencies(data[selected_column])
            translated_word_freq = get_word_frequencies(data[f'{selected_column}_translated'])
            
            st.subheader("Word Frequency Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Before Translation")
                st.write(dict(original_word_freq.most_common(10)))
            
            with col2:
                st.write("### After Translation")
                st.write(dict(translated_word_freq.most_common(10)))

        # Main display
        st.subheader("Processed Data Preview")
        st.dataframe(data.head())

        # Analysis sections
        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Topic Modeling", "Word Cloud", "Clustering"])

        with tab1:
            st.header("Sentiment Analysis")
            data['sentiment_score'] = data[selected_column].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            data['sentiment_category'] = data['sentiment_score'].apply(
                lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
            )
            st.write(data[[selected_column, 'sentiment_score', 'sentiment_category']])

        with tab2:
            st.header("Topic Modeling")
            vectorizer = CountVectorizer(stop_words='english')
            dtm = vectorizer.fit_transform(data[selected_column])
            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            lda.fit(dtm)

            for idx, topic in enumerate(lda.components_):
                st.subheader(f"Topic {idx+1}")
                st.write(", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]))

        with tab3:
            st.header("Word Cloud")
            all_text = " ".join(data[selected_column].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        with tab4:
            st.header("Text Clustering")
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(data[selected_column])
            kmeans = KMeans(n_clusters=3, random_state=42)
            data['cluster'] = kmeans.fit_predict(tfidf_matrix)
            st.write(data[[selected_column, 'cluster']])

        # Download section
        st.sidebar.header("Download Results")
        if st.sidebar.button("Prepare Download"):
            output = data.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="Download Processed Data",
                data=output,
                file_name='processed_data.csv',
                mime='text/csv'
            )

else:
    st.info("Please upload a file to begin analysis")
