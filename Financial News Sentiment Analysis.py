#!/usr/bin/env python
# coding: utf-8

# # Financial News Sentiment Analysis | Feature Engineering, EDA, and Modeling
In this project, our aim is to build a machine learning model capable of predicting whether financial news articles are positive or negative. Prior to creating the model, we'll conduct exploratory data analysis and perform feature engineering on the dataset to better understand and enhance the information contained in the financial news articles.
# Table of Contents
# 
# 1.Data Preprocessing
# .Importing Libraries
# .Reading in a dataset
# .Exploratory Data Analysis
# 
# 2.Feature Engineering
# .Processing for Missing Values and Outliers
# .Creating New Feature Interactions
# 
# 3.Modeling
# .Processing Encoding & One-Hot Encoding
# .Standardization for Numerical Variables
# .Create Modeling
# 
# 4.Summary

# # Install Liabraries:

# In[1]:


import pandas as pd
import numpy as np


# Data Visualization
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.offline as pyo
import seaborn as sns
from termcolor import colored
import plotly.express as px
import warnings


# Data Clustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Color palette for plots
colorpalt = ["#d62828", "#f77f00", "#fcbf49", "#003049"]

# Plotting setup
sns.set(style="darkgrid",color_codes=True)
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

k=KNeighborsClassifier()
d=DecisionTreeClassifier()
r=RandomForestClassifier()
l=LogisticRegression()
mb=MultinomialNB()
    
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')  


# In[2]:


get_ipython().system('pip install xgboost')


# # Load Dataset

# In[3]:


df = pd.read_csv("all-data.csv", encoding = "ISO-8859-1",  names=['sentiment', 'news'])
df.head()


# # Understanding the Data
# .Shape of the data
# 
# .Check column dtypes
# 
# .Check is there any null values

# In[4]:


# Check for missing data
df.isnull().sum()


# In[5]:


# dimensionality of the data
df.shape


# In[6]:


df.describe()


# # Checking Duplicates

# In[7]:


df[df.duplicated(['news'])]


# # Drop Duplicates

# In[8]:


df = df.drop_duplicates(subset={"sentiment", "news"}, keep='first', inplace=False)


# In[9]:


df_no_duplicates = df.drop_duplicates()


# In[10]:


df_dub_after_removal = df_no_duplicates.duplicated().any()
print("Are there duplicates after removal?", df_dub_after_removal)


# In[11]:


print("shape of the data :",df.shape)
print(colored('*'*42, 'blue'))

df.head()


# # Distribution of sentiment

# In[12]:


df['sentiment'].value_counts()


# In[13]:


plt.figure(figsize=(5, 3))
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()


# # Exploratory Data Analysis (EDA) on the required Features:

# In[14]:


pip install textblob


# In[15]:


from textblob import TextBlob

df['polarity'] = df['news'].map(lambda text: TextBlob(text).sentiment.polarity)
df['news_len'] = df['news'].astype(str).apply(len)
df['word_count'] = df['news'].apply(lambda x: len(str(x).split()))


# In[16]:


df


# In[17]:


print(df.columns)


# In[18]:


pip install wordcloud


# In[19]:


from wordcloud import WordCloud,STOPWORDS


# In[20]:


df1 = df[df['sentiment']=='positive']
words = ' '.join(df1['news'].astype(str))
cleaned_word = ' '.join([word for word in words.split() if not word.startswith('@')])

wordcloud = WordCloud(background_color='white',stopwords=STOPWORDS,
                      width=3000, height=2500).generate(''.join(cleaned_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[21]:


df2 = df[df['sentiment']=='negative']
words = ' '.join(df2['news'].astype(str))
cleaned_word = ' '.join([word for word in words.split() if not word.startswith('@')])

wordcloud = WordCloud(background_color='red',stopwords=STOPWORDS,
                      width=3000, height=2500).generate(''.join(cleaned_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[22]:


df3 = df[df['sentiment']=='neutral']
words = ' '.join(df3['news'].astype(str))
cleaned_word = ' '.join([word for word in words.split() if not word.startswith('@')])

wordcloud = WordCloud(background_color='black',stopwords=STOPWORDS,
                      width=3000, height=2500).generate(''.join(cleaned_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# # Drop Neutral Sentiment:

# In[23]:


df = df[df.sentiment != "neutral"]
df.head()


# In[24]:


df.info()


# # Sentiment Distribution

# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

sentiment = df['sentiment'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(x=sentiment.index, y=sentiment.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('sentiment', fontsize=12)
plt.xticks(rotation=90)
plt.show();


# In[26]:


plt.figure(figsize=(18, 12))
plt.subplot(221)
df["sentiment"].value_counts().plot.pie(autopct="%1.0f%%", colors=sns.color_palette("prism", 5), startangle=80, labels=["positive", "negative"], wedgeprops={"linewidth": 2, "edgecolor": "k"}, explode=[0.1, 0.1], shadow=True)
plt.title("Distribution")


# From the above plot positive news percentage is way too higher than negative news.

# # Data preparation

# ### Data Cleaning

# In[27]:


# from bs4 import BeautifulSoup

df["news"]=df["news"].str.lower() #We convert our texts to lowercase.
df["news"]=df["news"].str.replace("[^\w\s]","") #We remove punctuation marks from our texts.
df["news"]=df["news"].str.replace("\d+","") #We are removing numbers from our texts.
df["news"]=df["news"].str.replace("\n","").replace("\r","") #We remove spaces in our texts.

# def strip_html_tags(text):
#   soup = BeautifulSoup(text, "html.parser")
#   [s.extract() for s in soup(['iframe', 'script'])]
#   stripped_text = soup.get_text()
#   stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
#   return stripped_text

# def remove_accented_chars(text):
#   text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#   return text

# def stopwords_removal(words):
#     list_stopwords = nltk.corpus.stopwords.words('english')
#     return [word for word in words if word not in list_stopwords]


# In[28]:


df.head()


# In[29]:


#Summary satistics of numerical columns
df.describe(include=[np.number])


# In[30]:


#Summary of Categorical Columns
df.describe(include = 'object')


# # Define a Function to grab the Numerical and Categorical variables of its dataset

# In[31]:


def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# In[32]:


# Target Variable Analysis
df.sentiment.value_counts()


# In[33]:


num_cols


# In[34]:


def target_summary_with_num(dataframe,target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n")
    print("###################################")
    
for col in num_cols:
    target_summary_with_num(df,"sentiment",col)


# In[35]:


def target_density_est_with_num(dataframe,sentiment, numerical_col):
    plt.figure(figsize=(15,8))
    ax = sns.kdeplot(df[col][df.sentiment == 'positive'], color="green", shade=True)
    sns.kdeplot(df[col][df.sentiment == 'negative'], color="red", shade=True)
    plt.legend(['positive','negative'])
    plt.xlim(df[col].min(),df[col].max())
    plt.title("Sentiment Density of Numerical Variables")
    plt.show()
    
for col in num_cols:
    target_density_est_with_num(df,"sentiment",col)


# In[36]:


sns.pairplot(df, hue = 'sentiment' , vars=['polarity','news_len', 'word_count' ])


# In[37]:


df_numeric = df[[ 'polarity', 'news_len', 'word_count']]


# In[38]:


df_numeric.head()


# In[39]:


from scipy import stats

#Calculating Zscore of numeric columns in the dataset
z=np.abs(stats.zscore(df_numeric))
print (z)


# From these points it is very difficult to say which point is outliers so we will define threshold.
# 

# In[40]:


def outlier_th(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# In[41]:


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# In[42]:


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# In[43]:


for col in num_cols:
    print(col, check_outlier(df, col))


# In[44]:


for col in num_cols:
    replace_with_thresholds(df, col)


# In[45]:


for col in num_cols:
    print(col, check_outlier(df, col))


# # Correlation Analysis

# In[46]:


dimension_variable =["polarity", "news_len", "word_count"]
corr_matrix = df[dimension_variable].corr()
corr_matrix


# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,10)) # maximizing the size of graph
ax = sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", linewidths=.5, ax=ax)


# Step 6 : Splitting Data for Evaluation
# Split the Data in three Parts
# 
# -Train for Initial Selection
# -Val for Validation & Optimization

# # Handling imbalance (oversampling)

# In[48]:


from sklearn.utils import resample
# Separate majority and minority classes in training data for upsampling 
data_majority = df[df['sentiment'] == "positive"]
data_minority = df[df['sentiment'] == "negative"]

print("majority class before upsample:",data_majority.shape)
print("minority class before upsample:",data_minority.shape)

# Upsample minority class
data_minority_upsampled = resample(data_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= data_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_balance = pd.concat([data_majority, data_minority_upsampled])
 
# Display new class counts
print("After upsampling\n",df_balance.sentiment.value_counts(),sep = "")


# This code balances the dataset by upsampling the minority class (negative sentiment) to match the majority class (positive sentiment), ensuring equal representation for training a machine learning model on financial news sentiment analysis.

# In[49]:


# y = df["sentiment"]
# x = df.drop(["sentiment"], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)


# In[50]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.news, df.sentiment, test_size=0.1, random_state=42)
X_train.shape , X_test.shape, y_train.shape, y_test.shape


# # Checking distribution of target variable

# # 80% for training and 20% for testing

# In[51]:


print('Distribution of sentiment in training set')
print(y_train.value_counts())

print('Distribution of sentiment in test set')
print(y_test.value_counts())


# In[52]:


print('----------------Training Set-------')
print(X_train.shape)
print(y_train.shape)

print('----------------Test Set-------')
print(X_test.shape)
print(y_test.shape)


# # Modeling

# # Tokenizer

# In[53]:


from tensorflow.keras.preprocessing.text import Tokenizer

token = Tokenizer()
token.fit_on_texts(X_train)


# In[54]:


vocab = len(token.index_word) + 1
print("Vocabulary size={}".format(len(token.word_index)))
print("Number of Documents={}".format(token.document_count))


# # Sequence

# In[55]:


X_train = token.texts_to_sequences(X_train)
X_test = token.texts_to_sequences(X_test)


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')

train_lens = [len(s) for s in X_train]
test_lens = [len(s) for s in X_test]

fig, ax = plt.subplots(1,2, figsize=(12, 6))
h1 = ax[0].hist(train_lens)
h2 = ax[1].hist(test_lens)


# In[57]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

#padding
MAX_SEQUENCE_LENGTH = 30
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
X_train.shape, X_test.shape


# # Encoding Labels

# # Processing Encoding and One-Hot Encoding

# In[58]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
num_classes=2 # positive -> 1, negative -> 0


# In[59]:


y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# # Build Model

# In[60]:


import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D 

vec_size = 300
model = Sequential()
model.add(Embedding(vocab, vec_size, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(64,8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.1))

model.add(Dense(8, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.1))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
model.summary()


# # Train model

# In[61]:


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

epochs = 10
batch_size = 4

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('./model/sentiment_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train,  batch_size=batch_size, shuffle=True, validation_split=0.1, epochs=epochs, verbose=1, callbacks=[es, mc])


# # Evaluation

# # Model Accuracy

# In[62]:


from keras.models import load_model

saved_model = load_model('./model/sentiment_model.h5')
train_acc = saved_model.evaluate(X_train, y_train, verbose=1)
test_acc = saved_model.evaluate(X_test, y_test, verbose=1)
print('Train: %.2f%%, Test: %.2f%%' % (train_acc[1]*100, test_acc[1]*100))


# # Identify Overfitting

# In[63]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[64]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # Model Evaluation

# # Confusion Matrix

# In[65]:


def predictions(x):
    prediction_probs = model.predict(x)
    predictions = [1 if prob > 0.5 else 0 for prob in prediction_probs]
    return predictions


# In[66]:


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

labels = ['positive', 'negative']
print("Accuracy: %.2f%%" % (accuracy_score(y_test, predictions(X_test))*100))
print("Precision: %.2f%%" % (precision_score(y_test, predictions(X_test), average="macro")*100))
print("Recall: %.2f%%" % (recall_score(y_test, predictions(X_test), average="macro")*100))
print("F1_score: %.2f%%" % (f1_score(y_test, predictions(X_test), average="macro")*100))
print('================================================\n') 
print(classification_report(y_test, predictions(X_test)))
pd.DataFrame(confusion_matrix(y_test, predictions(X_test)), index=labels, columns=labels)


# # Report Analyzation :

# .Precision: Out of all instances predicted as positive, 89% were actually positive.
# .Recall: Out of all actual positive instances, the model correctly identified 98%.
# .F1-score: A balance between precision and recall, providing an overall measure of a model's accuracy.
#     
# Confusion Matrix Details:
#     
# .True Positive (TP): 48 instances were correctly predicted as positive.
# .False Positive (FP): 16 instances were predicted as positive but were actually negative.
# .True Negative (TN): 130 instances were correctly predicted as negative.
# .False Negative (FN): 3 instances were predicted as negative but were actually positive.

# # ROC AUC CURVE 

# In[68]:


def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[69]:


plot_roc_curve(y_test, predictions(X_test))
print("model AUC score: %.2f%%" % (roc_auc_score(y_test, predictions(X_test))*100))


# # 4. Summary :

# .The model performs very well with an accuracy of greater than 90%.
# 
# .High precision and recall for the positive class suggest a strong ability to correctly identify positive instances.
# 
# .The confusion matrix provides detailed insights into the model's predictions, indicating low false positive and false negative rates.
# 
# .In conclusion, this model seems to be highly effective in distinguishing between positive and negative instances, with a particularly high performance on the positive class.

# In[ ]:




