import pickle
import streamlit as st


def main():
    st.set_page_config(layout="wide")
    try:
        tfidf_loaded = pickle.load(open("tfidf_vectorizer.pickle.dat", "rb"))
        sentiment_model = pickle.load(open("logisticRegressionmodel.pickle.dat", "rb"))
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()

    # Function to predict the sentiment of the tweet (positive or negative)
    def predict_sentiment(tweet):
        # Transform the tweet using the TF-IDF vectorizer
        tweet_tfidf = tfidf_loaded.transform([tweet])

        # Convert the sparse matrix to a dense format
        tweet_tfidf_dense = tweet_tfidf.toarray()

        # Predict sentiment using the loaded model
        prediction = sentiment_model.predict(tweet_tfidf_dense)
        if prediction == 1:
            return st.success("😊 Positive: This tweet has a positive sentiment")
        else:
            return st.error("😞 Negative: This tweet has a negative sentiment.")

    st.title('Sentiment Analysis')
    st.markdown('---')

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.image('X.jpg')
    with col2:
        st.title('Twitter Sentiment Analyzer')

    st.markdown("---")
    st.subheader('Introduction')

    st.write(''' 
    This Sentiment Analysis App uses a machine learning model to classify tweets based on emotions
    as either positive or negative. The model is trained on a dataset of labeled tweets, 
    and after preprocessing (using TF-IDF), it learns to distinguish sentiment in text.
    When a user submits a tweet, the app predicts whether it has a positive or negative sentiment.
    ''')
    st.markdown("---")

    st.header('How Sentiment Analysis Works')

    st.write(''' 
    Sentiment analysis works by transforming text (tweets in this case) into feature vectors 
    using TF-IDF. The machine learning model is then trained to identify patterns in the text 
    that correspond to positive or negative sentiment.
    Once the model is trained, it predicts the sentiment of new tweets based on these learned patterns.
    ''')
    st.markdown("---")

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.image('emotions.jpg')
    with col2:
        st.header('Twitter Sentiment Analysis using Machine Learning')

    # User input for the tweet to analyze
    user_tweet = st.text_area('Enter the tweet to check sentiment:')

    # Button to trigger prediction
    if st.button('Analyze Sentiment'):
        if user_tweet:
            result = predict_sentiment(user_tweet)
            st.info(f'The tweet was analyzed using a machine learning  model.This is the emotions behind the tweet')
        else:
            st.write('Please enter a tweet to analyze.')
if __name__ == '__main__':
    main()