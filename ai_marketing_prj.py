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

import zipfile

# Path to your zip file
zip_file_path = 'final_data.zip'

# Name of the CSV file inside the zip file
csv_file_name = 'final_data.csv'

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as z:
    # Open the CSV file within the zip file
    with z.open(csv_file_name) as csv_file:
        # Read the CSV file into a DataFrame
        dftemp = pd.read_csv(csv_file)
        

# List of columns to drop
columns_to_drop = ['reviewerName', 'rank', 'date', 'fit', 'reviewerID', 'verified', 'unixReviewTime', 'feature']

# Drop the specified columns
dftemp = dftemp.drop(columns=columns_to_drop)
# Define the columns to check for missing values
columns_to_check = ['reviewText', 'summary', 'title']
# Drop rows with missing values in the specified columns
dftemp = dftemp.dropna(subset=columns_to_check)
# Replace missing values in the 'brand' column with "Others"
dftemp['brand'] = dftemp['brand'].fillna("Others")
    


# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load datasets
@st.cache_data
def load_data():
    df11 = pd.read_csv(r"sentimentdata.csv")
    df22 = dftemp.copy()
    return df11, df22

df11, df22 = load_data()

#ReviewTime Column
# Splitting the 'reviewTime' column
date_split = df22["reviewTime"].str.split(", ", n=1, expand=True)

# Splitting the date further into 'month' and 'day'
month_day_split = date_split[0].str.split(" ", n=1, expand=True)

# Adding 'year', 'month', and 'day' to the main dataset
df22["year"] = date_split[1]
df22["month"] = month_day_split[0]  # Extracting the month
df22["day"] = month_day_split[1]    # Extracting the day

# Dropping the original 'reviewTime' column
df22.drop(columns=["reviewTime"], inplace=True)

# Dropdown menus for brand, year, and month in a single row
col1, col2, col3 = st.columns(3)
with col1:
    brands = ['All'] + sorted(df22['brand'].unique().tolist())
    selected_brand = st.selectbox('Select Brand:', brands)

# Update year based on selected brand
with col2:
    if selected_brand != 'All':
        years = ['All'] + sorted(df22[df22['brand'] == selected_brand]['year'].unique().tolist())
    else:
        years = ['All'] + sorted(df22['year'].unique().tolist())
    selected_year = st.selectbox('Select Year:', years)

# Update month based on selected brand and year
with col3:
    if selected_brand != 'All' and selected_year != 'All':
        months = ['All'] + sorted(df22[(df22['brand'] == selected_brand) & (df22['year'] == selected_year)]['month'].unique().tolist())
    elif selected_year != 'All':
        months = ['All'] + sorted(df22[df22['year'] == selected_year]['month'].unique().tolist())
    else:
        months = ['All'] + sorted(df22['month'].unique().tolist())
    selected_month = st.selectbox('Select Month:', months)


# Applying filters to the data
# Filter by brand
if selected_brand != 'All':
    df1 = df11[df11['brand'] == selected_brand]
    df2 = df22[df22['brand'] == selected_brand]
else:
    df1 = df11.copy()
    df2 = df22.copy()

# Filter by year
if selected_year != 'All':
    df1 = df1[df1['year'] == selected_year]
    df2 = df2[df2['year'] == selected_year]

# Filter by month
if selected_month != 'All':
    df1 = df1[df1['month'] == selected_month]
    df2 = df2[df2['month'] == selected_month]

    
    
########Data Preprocessing

#change to round off instead
df2['overall'] = df2['overall'].apply(round)

#Figuring out the distribution of categories
overall_counts = df2['overall'].value_counts()

# Define a function to map ratings to sentiment categories
def map_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

# Create the sentiment column by applying the function to the 'overall' column
df2['sentiment'] = df2['overall'].apply(map_sentiment)


####Sentiment Analysis
def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Clean the reviewText column
df2['reviewText'] = df2['reviewText'].apply(review_cleaning)



#Spell Checker
from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


# Conducting sentiment analysis on reviewText
df2['sentiment_score_reviewText'] = df2['reviewText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df2['sentiment_reviewText'] = df2['sentiment_score_reviewText'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))

#####end of Sentiment Analysis

###Configuration
st.sidebar.header("Configuration")
dataset_choice = st.sidebar.selectbox("Choose the dataset for analysis:", ["Processed Data", "Raw Data"], index=0)

# Conditionally display the analysis type radio button
if dataset_choice == "Raw Data":
    analysis_type = st.sidebar.radio("Select analysis type:", ["Summary", "Raw Data Overview"])

    if analysis_type == "Summary":
        # Plot the distribution using a bar plot
        # Check the distribution of sentiment categories
        # Display summary numbers
        total_brands = df2['brand'].nunique()
        total_reviews = df2['reviewText'].count()
        st.write(f"**Summary Information**")
        st.write(f"Total Number of Brands: {total_brands}")
        st.write(f"Total Number of Reviews: {total_reviews}")

        # Plotting the overall rating distribution
        st.subheader("Overall Rating Distribution")
        plt.figure(figsize=(8, 6))
        df2['overall'].value_counts().sort_index().plot(kind='bar', color='blue')
        plt.title('Distribution of Overall Ratings')
        plt.xlabel('Ratings')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt)  # Display the plot

        # Plot the sentiment distribution using a bar plot
        st.subheader("Sentiment Distribution Based on Ratings")
        sentiment_counts = df2['sentiment'].value_counts()
        plt.figure(figsize=(8, 6))
        sentiment_counts.plot(kind='bar', color=['green', 'yellow', 'red'])
        plt.title('Sentiment Distribution based on Ratings')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt)  # Display the plot
        
    if analysis_type == "Raw Data Overview":
        st.header("Sample Data")
        st.dataframe(df2.head(20)) 


if dataset_choice == "Processed Data":
    analysis_type = st.sidebar.radio("Know your Customers", ["Sentiment Analysis", "Customer Feedback and Reviews"])

    if analysis_type == "Sentiment Analysis":
        # Positive reviews word cloud
        positive_reviews = df2[df2['sentiment_reviewText'] == 'positive']['reviewText']
        if not positive_reviews.empty:
            st.subheader("Positive Reviews Word Cloud")  # Adding a header for positive reviews
            wordcloud_text_positive = ' '.join(positive_reviews)
            wordcloud_positive = WordCloud(stopwords=STOPWORDS, background_color='white').generate(wordcloud_text_positive)
            fig_positive, ax_positive = plt.subplots()
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig_positive)
        else:
            st.subheader("Positive Reviews Word Cloud")  # Header displayed even if no reviews are available
            st.write("No positive reviews available for generating a word cloud.")

        # Negative reviews word cloud
        negative_reviews = df2[df2['sentiment_reviewText'] == 'negative']['reviewText']
        if not negative_reviews.empty:
            st.subheader("Negative Reviews Word Cloud")  # Adding a header for negative reviews
            wordcloud_text_negative = ' '.join(negative_reviews)
            wordcloud_negative = WordCloud(stopwords=STOPWORDS, background_color='black', contour_color='white').generate(wordcloud_text_negative)
            fig_negative, ax_negative = plt.subplots()
            plt.imshow(wordcloud_negative, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig_negative)
        else:
            st.subheader("Negative Reviews Word Cloud")  # Header displayed even if no reviews are available
            st.write("No negative reviews available for generating a word cloud.")

    if analysis_type == "Customer Feedback and Reviews":
        st.subheader("Distribution of Sentiment Scores")
        # Plotting histogram of sentiment scores from review text
        plt.figure(figsize=(10, 4))
        plt.hist(df2['sentiment_score_reviewText'], bins=20, color='purple', alpha=0.7)
        plt.title("Histogram of Sentiment Scores from Review Text")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        st.pyplot(plt)
    
        # Aggregate the average sentiment scores per brand and display top 10 and bottom 10 brands
        brand_sentiment_avg = df2.groupby('brand')['sentiment_score_reviewText'].mean().sort_values()
    
        st.subheader("Top 10 Brands with Positive Reviews")
        top_10_brands = brand_sentiment_avg.tail(10)
        plt.figure(figsize=(10, 4))
        top_10_brands.plot(kind='barh', color='green')
        plt.title("Top 10 Brands by Average Sentiment Score")
        plt.xlabel("Average Sentiment Score")
        st.pyplot(plt)
    
        st.subheader("Bottom 10 Brands with Negative Reviews")
        bottom_10_brands = brand_sentiment_avg.head(10)
        plt.figure(figsize=(10, 4))
        bottom_10_brands.plot(kind='barh', color='red')
        plt.title("Bottom 10 Brands by Average Sentiment Score")
        plt.xlabel("Average Sentiment Score")
        st.pyplot(plt)
    
        # Comparison of sentiment counts based on rating vs review text
        st.subheader("Comparison of Sentiment Counts: Rating vs Review Text")
        comparison_df = pd.DataFrame({
            'Sentiment from Ratings': df2['sentiment'].value_counts(),
            'Sentiment from Review Text': df2['sentiment_reviewText'].value_counts()
        })
    
        plt.figure(figsize=(10, 5))
        comparison_df.plot(kind='bar', color=['skyblue', 'orange'])
        plt.title("Sentiment Comparison: Ratings vs Review Text")
        plt.xlabel("Sentiment Type")
        plt.ylabel("Counts")
        plt.xticks(rotation=0)
        st.pyplot(plt)
