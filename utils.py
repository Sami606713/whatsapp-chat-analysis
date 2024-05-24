import pandas as pd
import re
import numpy as np
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urlextract
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def fun_to_get_sentiment(df,text):
    temp=df[df['user']!='invitation']
    courpus=temp[temp['sentiment']==text]['message'].tolist()
    return " ".join(courpus)
# Function to process the txt file and return dataframe
def sentiment(text_score):
    if(text_score>=0.05):
        return "positive"
    elif(text_score<=-0.05):
        return "negative"
    else:
        return "neutral"
    
def preprocess(data):
    # Pattern for extract date
    pattern=r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}"
    date=re.findall(pattern,data)
    
    pattern = r'\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M - (.*)'

    # Extracting the text using findall method
    messages = re.findall(pattern, data)
    df=pd.DataFrame({"date":date,"mess":messages})
    
    # Saperate the user name and message
    user=[]
    message=[]
    for mess in df['mess']:
        entry=re.split('([\w\W]+?):\s',mess)
        if(entry[1:]):
            user.append(entry[1])
            message.append(entry[2])
    #         print(entry[1:])
        else:
            user.append("invitation")
            message.append(entry[0])
    
    df['user']=user
    df['message']=message
    
    # Convert date to datetime
    df['date']=pd.to_datetime(df['date'])
    
    # Extract the year month day,hour and min
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month_name()
    df['day']=df['date'].dt.day_name()
    df['hour']=df['date'].dt.hour
    df['min']=df['date'].dt.minute
    
    # Apply the vardar model for sentiment
    analyser=SentimentIntensityAnalyzer()
    df['sentiment']=df['message'].apply(lambda x:analyser.polarity_scores(x)['compound'])
    df['sentiment']=df['sentiment'].apply(sentiment)

    # Drop the mess col
    df.drop(columns=['mess'],inplace=True)
    # print(df.head(3))
    return df

def procecss_data_frame(df):
    unique_users = np.unique(df['user'].values)
    unique_users_with_all = np.insert(unique_users, 0, "OverAll")
    option=st.selectbox("",unique_users_with_all)
    
    if option=="OverAll":
        return df
    else:
        df=df[df['user']==option]
        return df

def get_url(text):
    extract=urlextract.URLExtract()
    urls=extract.find_urls(text)
    if (len(urls)==0):
        return 0
    else:
        return len(urls)
    return urls

def monthly_stat(df):
    df=df[df['user']!='invitation']
    temp = df.groupby('month')['user'].value_counts().reset_index()
    plt.figure(figsize=(15,7))
    plt.title("Monthly Time-line")
    plt.plot(temp['month'],temp['count'],color='red',marker="o")
    plt.xlabel("Month")
    plt.ylabel('Count')
    # plt.grid(True)
    st.pyplot(plt)

# Most active user
def barplot(df):
    df=df[df['user']!='invitation']
    # Get the unique values of year
    unique_year=np.unique(df['year'].values).astype(str)
    # insert over all at 0 index
    yeas_with_all=np.insert(unique_year, 0, "OverAll")
    year=st.selectbox("select year",yeas_with_all)

    if (year=="OverAll"):
        temp=df['user'].value_counts().head(5).reset_index()
    else:
        df=df[df['year']==int(year)]
        temp=df['user'].value_counts().head(5).reset_index()
    plt.figure(figsize=(10,7))
    plt.title("Monthly Time-line")
    sns.barplot(x=temp['user'],y=temp['count'],palette='viridis')
    plt.xlabel("User")
    plt.ylabel('Count')
    # plt.grid(True)
    st.pyplot(plt)


def daily(df):
    temp=df.groupby(['user'])['day'].value_counts().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(15,7))
    plt.title("Most Bussy Day")
    sns.barplot(x=temp['day'],
                y=temp['count'],errorbar=None,color="yellow")
    st.pyplot(plt)

def monthly(df):
    temp=df.groupby(['user'])['month'].value_counts().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(15,7))
    plt.title("Most Bussy Month")
    sns.barplot(x=temp['month'],
                y=temp['count'],errorbar=None,palette='winter')
    st.pyplot(plt)


# display word cloud
def word_cloud(df):
    option=st.selectbox("",np.unique(df['sentiment'].values))
    text=fun_to_get_sentiment(df=df,text=option)
    cloud=WordCloud(background_color='white',width=800, height=400).generate(text=text)
    plt.figure(figsize=(8,6))
    plt.title(f"{option} text")
    plt.imshow(cloud)
    st.pyplot(plt)
