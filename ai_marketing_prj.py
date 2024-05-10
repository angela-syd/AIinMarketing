pip install matplotlib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS


st.set_page_config(layout="wide")  # change layout

# trying fade-in effect
title_html = """
    <style>
        @keyframes fadeIn {{
            0% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}
        .title {{
            text-align: center;
            animation-name: fadeIn;
            animation-duration: 2s;
        }}
    </style>
    <h1 class="title">Brand Sentiment Analysis Dashboard</h1>
"""

st.markdown(title_html, unsafe_allow_html=True)

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load datasets
@st.cache_data
def load_data():
    df1 = pd.read_csv(r"C:\Users\X1 Carbon\Downloads\sentimentdata.csv")
    df2 = pd.read_csv(r"C:\Users\X1 Carbon\Downloads\final_data.csv")
    return df1, df2

df1, df2 = load_data()

# Function to preprocess text and perform sentiment analysis
def preprocess_and_analyze(df):
    df['cleanedText'] = df['reviewText'].apply(lambda text: re.sub('https?://\S+|www\.\S+', '', str(text).lower()))
    df['cleanedText'] = df['cleanedText'].apply(lambda text: re.sub('[%s]' % re.escape(string.punctuation), '', text))
    df['sentiment_score'] = df['cleanedText'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_type'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))

preprocess_and_analyze(df1)

# Sidebar configuration
st.sidebar.header("Configuration")
dataset_choice = st.sidebar.selectbox("Choose the dataset for analysis:", ["Processed Data", "Raw Data"], index=0)
#analysis_type = st.sidebar.radio("Select analysis type:", ["Sentiment Analysis", "Data Overview"])

# Conditionally display the analysis type radio button
if dataset_choice == "Processed Data":
    analysis_type = st.sidebar.radio("Select analysis type:", ["Sentiment Analysis", "Data Overview"])

    if analysis_type == "Sentiment Analysis":

        st.write("Sentiment Analysis Details here...")   #figure out how to make it dynamic



if dataset_choice == "Processed Data":
    if analysis_type == "Sentiment Analysis":
        positive_reviews = df1[df1['sentiment_type'] == 'positive']['cleanedText']
        if not positive_reviews.empty:
            wordcloud_text = ' '.join(positive_reviews)
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(wordcloud_text)
            fig, ax = plt.subplots()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig)
        else:
            st.write("No positive reviews available for generating a word cloud.")

# Function to preprocess text
def preprocess_text(text):
    text = str(text).lower() 
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


# Display raw data selection
def display_data(data):
    st.dataframe(data.head())


def plot_data(df, title):
    fig, ax = plt.subplots()
    df['overall'].plot(kind='hist', bins=20, title=title)
    plt.gca().spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)


dataset_choice = st.sidebar.radio("Know your Customers", ["Sentiment Analysis", "Customer Feedback and Reviews"])


if dataset_choice == "Sentiment Analysis":
    st.header("Sample Data")
    st.dataframe(df1.head(20))  

elif dataset_choice == "Customer Feedback and Reviews":
    st.header("Sample Data")
    st.dataframe(df2.head(20)) 


col1, col2 = st.columns(2)

# Plotting in the first column
with col1:
    st.header("Overall Rating Distribution")
    fig1, ax1 = plt.subplots()
    df1['overall'].plot(kind='hist', bins=20, title='Distribution of Overall Ratings - ' + dataset_choice)
    ax1.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig1)
    


logger = logging.getLogger(__name__)
logger.info("Application started successfully")

