import streamlit as st
import numpy as np
from utils import preprocess,procecss_data_frame,get_url,monthly_stat,barplot,daily,monthly,word_cloud
from nltk import word_tokenize,sent_tokenize

# Page config
# Set page title and favicon
st.set_page_config(
    page_title="WhatsApp Chat Analysis",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Define the custom CSS style for the title
title_style = """
    <style>
        .title-text {
            color:green;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
        }
    </style>
"""
# Display the styled title using markdown
st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<p class='title-text'>WhatsApp Chat Analysis</p>", unsafe_allow_html=True)


# Sidebar
df=None
with st.sidebar.title("Options"):
    st.header("File uploader")

    file=st.file_uploader("Upload file",type="txt")

    # Encode the file
    if file is not None:
        file_byte=file.getvalue()
        data=file_byte.decode('utf-8')
        df=preprocess(data)
        df=procecss_data_frame(df)

if df is not None:
    st.dataframe(df)

    st.header("Group Static")
    with st.container(border=True):
        col1,col2,col3,col4,col5=st.columns(5)
       
        with col1:
            with st.container(border=True,height=120):
                st.write("*Total Messages*")
                st.write(df.shape[0])
        with col2:
            with st.container(border=True,height=120):
                st.write("Total Words")
                word=np.cumsum(df['message'].apply(lambda x:len(word_tokenize(x)))).tail(1).values[0]
                st.write(word)
        with col3:
            with st.container(border=True,height=120):
                st.write("Total sentence")
                sent=np.cumsum(df['message'].apply(lambda x:len(sent_tokenize(x)))).tail(1).values[0]
                st.write(sent)
        with col4:
            with st.container(border=True,height=120):
                st.write("Total Media file")
                media=df[df['message']=='<Media omitted>'].shape[0]
                st.write(media)
        with col5:
            with st.container(border=True,height=120):
                st.write("Total Links")
                links=np.cumsum(df['message'].apply(get_url)).tail(1).values[0]
                st.write(links)

    # Monthly statics
    st.header("Monthly Stats")
    with st.container(border=True):
        monthly_stat(df)
    
    # Most bussy user
    st.header("Most Acive/Busy User")
    with st.container(border=True):
        barplot(df)
    
    st.header("Daily Activity")
    with st.container(border=True):
       daily(df)

    st.header("Monthly Activity")
    with st.container(border=True):
        monthly(df)
    

    st.header("Word Cloud")
    with st.container(border=True):
        word_cloud(df)
