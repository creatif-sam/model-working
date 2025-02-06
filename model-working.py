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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import spacy
from sentence_transformers import SentenceTransformer
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Page configuration
st.set_page_config(page_title="Trans-Sustain Analysis", layout="wide")

# Load models with caching
@st.cache_resource
def load_translation_model():
    return MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en"), \
           MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

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

        # Language detection and translation
        st.header("Language Detection and Translation")
        data['detected_lang'] = data[selected_column].apply(lambda x: detect(x) if pd.notnull(x) else "unknown")
        
        if "fr" in data['detected_lang'].values:
            tokenizer, model = load_translation_model()
            
            def translate_text(text):
                if pd.notnull(text) and detect(text) == "fr":
                    translated = tokenizer.decode(
                        model.generate(**tokenizer(text, return_tensors="pt", truncation=True))[0],
                        skip_special_tokens=True
                    )
                    return translated
                return text

            with st.spinner('Translating content...'):
                data[f'{selected_column}_translated'] = data[selected_column].apply(translate_text)
            st.success("Translation completed!")

            # Calculate translation percentage
            def translation_percentage(original, translated):
                if pd.notnull(original) and pd.notnull(translated) and detect(original) == "fr":
                    original_words = len(original.split())
                    translated_words = len(translated.split())
                    return min(100, (translated_words / original_words) * 100)
                return 100

            data['translation_percentage'] = data.apply(
                lambda row: translation_percentage(row[selected_column], row[f'{selected_column}_translated']), axis=1
            )

            # Display translation overview
            st.subheader("Translation Overview")
            st.dataframe(data[[selected_column, f'{selected_column}_translated', 'translation_percentage']].head())

        # Analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sentiment Analysis", "Topic Modeling", "Word Cloud", "Clustering", "Cluster Evaluation"])

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
            
            # Dynamic cluster handling
            n_samples = len(data)
            safe_clusters = min(4, max(1, n_samples-1))  # Default to 4, but adjust for small datasets
            
            try:
                kmeans = KMeans(n_clusters=safe_clusters, random_state=42)
                data['cluster_kmeans'] = kmeans.fit_predict(tfidf_matrix)
                st.write(data[[selected_column, 'cluster_kmeans']])
            except Exception as e:
                st.error(f"Clustering failed: {str(e)}")

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
