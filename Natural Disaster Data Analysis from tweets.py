#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[9]:


#downloading packages
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix


# In[10]:


#Creating datasets
train_df = pd.read_csv('Desktop/train.csv')
test_df = pd.read_csv('Desktop/test.csv')


# In[11]:


#printing train data table
train_df.head()


# In[12]:


#Analyzing train data 
train_df.info()


# In[13]:


#isnull
train_df.isnull().sum()


# In[14]:


#duplicated
train_df.duplicated().sum()


# In[15]:


#value counts
train_df['location'].value_counts()


# In[16]:


#Replacing the 'USA' entry names with 'United States'
train_df['location'] = train_df['location'].replace(['USA'], 'United States')


# In[17]:


train_df['location'].value_counts()


# In[18]:


#reading tweets
train_df[train_df['target'] == 0]['text'][:10]


# In[19]:


#reading tweets
train_df[train_df['target'] == 1]['text'][:10]


# In[20]:


#creating plot where 0 = No Disaster and 1 = Disaster
sns.countplot(x = 'target', data = train_df)


# In[21]:


##Creating pie chart
#Counts the amount of disasters and non-disasters from 'target'. 
sum_disaster = train_df['target'].value_counts()
#Adds color to pie chart
colors = sns.color_palette()[0:2]
label_values = ['No Disaster', 'Disaster']
#Plots pie chart
plt.pie(sum_disaster.values, labels = label_values, colors = colors, autopct = '%.0f%%', shadow = True)
plt.title('Percentage of Count Values for Each Category')
#Edit font size
plt.rcParams['font.size'] = 12


# In[22]:


##Creating bar graphs of word count
#New data frames are created, with one containing all of the non-disaster (normal) tweets and another containing all of the disaster tweets.
normal_tweet = train_df[train_df['target'] == 0]
disaster_tweet = train_df[train_df['target'] == 1]
#A histogram plot for word length is created for both normal and disaster tweets and combined on one plot.
plt.hist(normal_tweet['text'].apply(lambda x: len(x.split())), label = 'Normal', bins = 15, alpha = 0.7)
plt.hist(disaster_tweet['text'].apply(lambda x: len(x.split())), label = 'Disaster', bins = 15, alpha = 0.7)
#The title, legend, and labels are created for the histogram plot.
plt.title('Total Word Amount in Tweets for Disasters and Non-Disasters')
plt.legend()
plt.xlabel('Number of Words')
plt.ylabel('Total Count')


# In[23]:


##Creating bar graphs of character count
plt.hist(normal_tweet['text'].apply(lambda x: len(x)), label = 'No Disaster', bins = 15, alpha = 0.7)
plt.hist(disaster_tweet['text'].apply(lambda x: len(x)), label = 'Actual Disaster', bins = 15, alpha = 0.7)

plt.title('Total Character Amount in Tweets for Disasters and Non-Disasters')
plt.legend()
plt.xlabel('Number of Characters')
plt.ylabel('Total Count')


# In[24]:


#Creating the set of stopwords in the English language.
stop = set([w for w in nltk.corpus.stopwords.words('english')])


# In[25]:


#Creating the text_cleaner function, which will be used to clean the text in the train_df dataframe.
def text_cleaner(text):
  #text = re.sub(r'MH370', 'Malaysia Airlines Flight 370', text) #Replaces MH370 with airline and flight number
  text = re.sub(r'http\S*\s?', '', text) #Removes URLs
  text = re.sub(r'\d+', '', text) #Removes numbers
  text = re.sub(r'[^\w\s]','', text) #Removes punctuation
  text = re.sub(r'http', '', text) #Removes http
  text = re.sub(r'wan', '', text) #Removes 'wan'
  text = re.sub(r'gon', '', text) #Removes 'gon'
  text = re.sub(r'&gt', '>', text) #Replaces gt with greater than symbol
  text = re.sub(r'&lt', '<', text) #Replaces lt with less than symbol
  text = re.sub(r'&amp;', '&', text) #Replaces &amp with ampersand symbol
  text = re.sub(r'\s+', ' ', text) #Removes extra whitespace
  text = ' '.join([i for i in text.split() if i not in stop])
  text = text.lower() #Makes all of the words lower case
  return text


# In[26]:


# Referencing: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emojis = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emojis.sub(r'', text)


# In[27]:


#Calls the WordNetLemmetizer from NLTK library and saves in the lemmatizer variable.
lemmatizer = nltk.stem.WordNetLemmatizer()
#Creates the lemmatize function.
def lemmatize_text(text):
  text = ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))
  return text


# In[28]:


#The text cleaner and remove emoji functions are applied to the train_df data frame.
train_df['text'] = train_df['text'].apply(lambda x: text_cleaner(x))
train_df['text'] = train_df['text'].apply(lambda x: remove_emoji(x))
train_df['text'] = train_df['text'].apply(lambda x:lemmatize_text(x))
#The normal and disaster tweets data frames are set up.
normal_tweet = train_df[train_df['target'] == 0]
disaster_tweet = train_df[train_df['target'] == 1]


# In[29]:


normal_tokens = [w for c in normal_tweet['text']
                    for s in nltk.sent_tokenize(c)
                    for w in nltk.word_tokenize(s)
                    if w.isalpha() and w not in stop]

normal_bigrams = nltk.FreqDist(nltk.bigrams(normal_tokens))
normal_bigrams


# In[30]:


disaster_tokens = [w for c in disaster_tweet['text']
                    for s in nltk.sent_tokenize(c)
                    for w in nltk.word_tokenize(s)
                    if w.isalpha() and w not in stop]

disaster_bigrams = nltk.FreqDist(nltk.bigrams(disaster_tokens))
disaster_bigrams


# In[31]:


#Creating a counter for the disaster bigrams
freq_count_normal = Counter(normal_bigrams)
#Creating a new Pandas dataframe of the count
normal_count = pd.DataFrame(freq_count_normal.most_common(20))
#Creating a barplot of the top 20 Bigrams of Disaster Tweets
plt.figure(figsize=(12,6))
norm_plot = sns.barplot(x = normal_count[1], y = normal_count[0])
#Sets the labels in the barplot.
norm_plot.set(xlabel = 'Count', ylabel = 'Bigram', title = 'Top 20 Bigrams of Normal Tweets')


# In[32]:


#Creating a counter for the disaster bigrams
freq_count_disaster = Counter(disaster_bigrams)
#Creating a new Pandas dataframe of the count
disaster_count = pd.DataFrame(freq_count_disaster.most_common(20))
#Creating a barplot of the top 20 Bigrams of Disaster Tweets
plt.figure(figsize=(12,6))
dis_plot = sns.barplot(x = disaster_count[1], y = disaster_count[0])
#Sets the labels in the barplot.
dis_plot.set(xlabel = 'Count', ylabel = 'Bigram', title = 'Top 20 Bigrams of Disaster Tweets')


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer #For Count Vectorizor.
from sklearn.model_selection import train_test_split #For train/test split.
from sklearn.linear_model import LogisticRegression #For Logistic Regression.
from sklearn.svm import SVC #For Support Vector Machine.
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


# In[34]:


#X and y are created to return a NumPy array. 
X = train_df['text'].values
y = train_df['target'].values


# In[35]:


#Train and test data set created.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[36]:


#Creating a pipeline for the Logistic Regression algorithm using Count Vectorizor and Tf-Idf.
pipeline_log = Pipeline([
    ('vect', CountVectorizer()),  #strings to token integer counts
    ('tfidf', TfidfTransformer()),  #integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression()),  #train on TF-IDF vectors w/ Logistic Regression
])


# In[37]:


pipeline_log.fit(X_train, y_train)

pipe_predict_log = pipeline_log.predict(X_test)


# In[38]:


#Printing the confusion matrix and classification report for the logreg predictions.
print(confusion_matrix(y_test, pipe_predict_log))
print('\n')
print(classification_report(y_test, pipe_predict_log))


# In[40]:


pipeline_mb = Pipeline([
    ('vect', CountVectorizer()),  #strings to token integer counts
    ('tfidf', TfidfTransformer()),  #integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  #train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[41]:


pipeline_mb.fit(X_train, y_train)


# In[42]:


pipe_predict_mb = pipeline_mb.predict(X_test)


# In[43]:


print(confusion_matrix(y_test, pipe_predict_mb))
print('\n')
print(classification_report(y_test, pipe_predict_mb))


# In[44]:


#Creating confusion matrix plot.
plot_confusion_matrix(pipeline_mb, X_test, y_test)
#Removing grid lines to clearly see the labels.
plt.grid(False)


# In[45]:


test_df['text'] = test_df['text'].apply(lambda x: text_cleaner(x))
test_df['text'] = test_df['text'].apply(lambda x: remove_emoji(x))
test_df['text'] = test_df['text'].apply(lambda x:lemmatize_text(x))


# In[46]:


#Creating test table
test_df.head()


# In[ ]:




