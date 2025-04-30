import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components
def main():
    st.set_page_config(layout="wide")
    st.title('How The Twitter Sentiment Analyser Was Built')
    st.markdown('---')
    st.write('''
    #### This is a notebook with the EDA,Data Preprocessing and Model Building
    ''')
    st.write('''
    This sentiment analyzer was created using machine learning. 
    First, a dataset of tweets labeled as positive or negative was collected. 
    The text was cleaned and processed to remove unnecessary characters. 
    Then,TF-IDF was used to turn the text into numbers so the computer could understand it. 
    After that, a machine learning model was trained to learn the patterns in the tweets. 
     ''')
    html_file=Path('Twitter_sentiment_Analysis.html')
    components.html(html_file.read_text(encoding='utf-8',errors='replace'),height=1000,scrolling=True)
if __name__ == '__main__':
    main()