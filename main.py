import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

st.markdown(
    """
    <style>
    body {
        background-color: #006400; /* Dark green color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Scrapping of data
def scrape_data():
    l = []
    for i in range(1, 30):
        url = "https://www.moneycontrol.com/news/tags/cryptocurrency.html/page-{}/".format(i)
        try:
            html = requests.get(url)
        except requests.exceptions.ContentDecodingError as e:
            print("Content decoding error:", str(e))
            continue

        bsobj = BeautifulSoup(html.content, "html.parser")
        for link in bsobj.find_all('h2'):
            l.append([link.text])

    data = pd.DataFrame(l)
    data.to_csv('data.csv')

    return data


# Text Cleaning
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# Sentiment Analysis using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity

    if sentiment_polarity > 0:
        return 'Positive'
    elif sentiment_polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'


def main():
    st.title("Cryptocurrency News Sentiment Analysis")

    # Scrape data
    st.subheader("Data Scraping")
    data = scrape_data()
    st.text("Data scraped successfully.")

    # Preprocess data
    st.subheader("Data Preprocessing")
    new_data = data.copy()
    new_data['Text'] = new_data[0].apply(preprocess_text)
    st.text("Data preprocessed successfully.")

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    coin = st.text_input("Enter a cryptocurrency keyword (e.g., bitcoin,ethereum,tether,xrp):")
    if coin:
        st.write(f"Sentiment analysis for {coin}:")
        filtered_data = new_data[new_data['Text'].str.contains(coin, case=False)]
        if not filtered_data.empty:
            filtered_data['Sentiment'] = filtered_data['Text'].apply(get_sentiment)
            st.dataframe(filtered_data)
        else:
            st.text("No news found for the given keyword.")
    else:
        st.text("Enter a keyword to perform sentiment analysis.")


if __name__ == '__main__':
    main()
