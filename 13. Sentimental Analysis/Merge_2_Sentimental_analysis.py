#-----------SENTIMENTAL ANALYSIS------------

#---->Sentiment analysis refers to analyzing an opinion or feelings about something using data 
      #like text or images, regarding almost anything. Sentiment analysis helps companies in  
      #their decision-making process. For instance, if public sentiment towards a product is 
      #not so good, a company may try to modify the product or stop the production altogether 
      #in order to avoid any losses.
     
#---->There are many sources of public sentiment e.g. public interviews, 
      #opinion polls, surveys, etc. However, with more and more people joining 
      #social media platforms, websites like Facebook and Twitter can be parsed 
      #for public sentiment.

#---->Here, we will see how we can perform sentiment analysis of text data of tweets.    

#---->The ability to categorize opinions expressed in the text of tweets—and especially to 
      #determine whether the writer's attitude is positive, negative, or neutral—is highly 
      #valuable.

#---->There are different ordinal scales used to categorize tweets. 
      #A five-point ordinal scale includes five categories: Highly Negative, Slightly Negative,
      #Neutral, Slightly Positive, and Highly Positive. A three-point ordinal scale 
      #includes Negative, Neutral, and Positive; and a two-point ordinal scale 
      #includes Negative and Positive.
     
#---->Here the 2 dataset are used for training and 'test.csv'f for testing. 
     #In the folder all the dataset are of 3-point
     #(Neutral,Negative,Positive)  

#-----------------1.LOADING THE DATASET --------------
import pandas as pd
import numpy as np

df2=pd.read_csv('tweet_3-point.csv')#(14640,15)
df3=pd.read_table('tweet1_3-point.tsv')#(7589,3) #0=Negative,1=Positive,-1=Neutral

#---------------2.DATA ANALYSIS ----------------

import warnings
warnings.filterwarnings("ignore")

#Here we have 3 dataset, so lets count total number of positive texts, negative texts and 
#neutral texts

#NEGATIVE TEXTS
neg_2=sum(df2['airline_sentiment']=='negative')
neg_3=sum(df3['label']==0)

#POSITIVE TEXTS
pos_2=sum(df2['airline_sentiment']=='positive')
pos_3=sum(df3['label']==1)

#NEUTRAL TEXTS
neu_2=sum(df2['airline_sentiment']=='neutral')
neu_3=sum(df3['label']==-1)

print('FOR tweet_3-point DATASET')
print('Positve:{0}\nNegative:{1}\nNeutral:{2}\n'.format(pos_2,neg_2,neu_2))

print('FOR tweet1_3-point DATASET')
print('Positve:{0}\nNegative:{1}\nNeutral:{2}\n'.format(pos_3,neg_3,neu_3))

print("TOTAL POSITIVE TWEETS:{0}\n".format(pos_2+pos_3))
print("TOTAL NEGATIVE TWEETS:{0}\n".format(neg_2+neg_3))
print("TOTAL NEUTRAL TWEETS:{0}\n".format(neu_2+neu_3))

#VISUALISATION USING PIE CHART
import matplotlib.pyplot as plt

df2.airline_sentiment.value_counts().plot(kind='pie',autopct='%1.0f%%',shadow=True,
            colors=['red','lightgreen','cyan'],labels=['Negative','Positive','Neutral'])
plt.show()

df3.label.value_counts().plot(kind='pie',autopct='%1.0f%%',shadow=True,
                              colors=['red','lightgreen','cyan'],labels=['Negative','Positive','Neutral'])
plt.show()

#------------------3.DATA CLEANING --------------------

a=pd.concat([df2['text'],df3['tweet']],ignore_index=True)
X=pd.DataFrame(a,columns=['text'])
print(len(X['text']))

Y=pd.concat([df2['airline_sentiment'],df3['label']],ignore_index=True)
Y_train_tweets=pd.DataFrame(Y,columns=['Sentiments'])
#Replacing old senitments by new classification values where 0=Neutral,1=Positive,2=Negative
Y_train_tweets['Sentiments']=Y_train_tweets['Sentiments'].replace(['neutral','positive','negative',-1,1,0],[0,1,2,0,1,2])


import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
clean_tweet_texts_of_train_dataset=[]


def data_cleaning_or_pre_processing(tweet,clean_tweet_texts):
    #Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove all the special characters
    tweet= re.sub(r'\W', ' ', tweet)
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove single characters from the start
    tweet= re.sub(r'\^[a-zA-Z]\s+', ' ', tweet)
    tweet=tweet.lower()
    tweet=tweet.split()
    
    stemming=PorterStemmer()
    tweet=[stemming.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet=' '.join(tweet)
    
    lemmatizer=WordNetLemmatizer()
    tweet=[lemmatizer.lemmatize(word) for word in tweet]
    tweet=''.join(tweet)
    
    clean_tweet_texts.append(tweet)
    
total_dataset_texts= 14640 + 7589 #1599999 +
for i in range(total_dataset_texts):
    data_cleaning_or_pre_processing(X['text'][i],clean_tweet_texts_of_train_dataset)

#Represting Text in numeric form Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
X_tweets= vectorizer.fit_transform(clean_tweet_texts_of_train_dataset).toarray()


#-----------------4. MODEL BUILDING -----------------------
global index
index=1

def model_building(classifier,X_train,Y_train,X_test,Y_test,name):

      classifier.fit(X_train,Y_train)

      Y_pred=classifier.predict(X_test)

      #Model Report
      """      
      There are four ways to check if the predictions are right or wrong:

          TN / True Negative: when a case was negative and predicted negative
          TP / True Positive: when a case was positive and predicted positive
          FN / False Negative: when a case was positive but predicted negative
          FP / False Positive: when a case was negative but predicted positive
          
          ----> 1. Precision – What percent of your predictions were correct?
          Precision is the ability of a classifier not to label an instance positive that is 
          actually negative. For each class it is defined as the ratio of true positives to the 
          sum of true and false positives.
          
          TP – True Positives
          FP – False Positives
          
          Precision – Accuracy of positive predictions.
          Precision = TP/(TP + FP)
          
          
          ----->2. Recall – What percent of the positive cases did you catch? 
          Recall is the ability of a classifier to find all positive instances. 
          For each class it is defined as the ratio of true positives to the sum of true positives 
          and false negatives.
          
          FN – False Negatives
          
          Recall: Fraction of positives that were correctly identified.
          Recall = TP/(TP+FN)
          
          -----> 3. F1 score – What percent of positive predictions were correct? 
          The F1 score is a weighted harmonic mean of precision and recall such that the best 
          score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy 
          measures as they embed precision and recall into their computation. As a rule of thumb, 
          the weighted average of F1 should be used to compare classifier models, not global accuracy.
          
          F1 Score = 2*(Recall * Precision) / (Recall + Precision)
          
          
          """     
      global index
      print("-----------{0}. {1}---------\n".format(index,name))
      index+=1
      from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
      print(confusion_matrix(Y_test,Y_pred))
      print("\nClassification Report")
      print(classification_report(Y_test,Y_pred))
      print("\n Accuracy of Model")
      print(round(accuracy_score(Y_test,Y_pred),2)*100)


#Applying test.csv
df_test=pd.read_table('test.tsv')
total_test_dataset_texts=1317
Y_test_tweets=df_test['label'].replace([-1,1,0],[0,1,2])
clean_tweet_texts_of_test_dataset=[]
for i in range(total_test_dataset_texts):
    data_cleaning_or_pre_processing(df_test['tweet'][i],clean_tweet_texts_of_test_dataset)

test_tweets_matrix = vectorizer.fit_transform(clean_tweet_texts_of_test_dataset).toarray()
#(22229, 2500)--->X_tweets,(1317, 511)-->Test_tweets
zero_array=np.zeros([1317,1989])
new_test_tweets_matrix=np.concatenate((test_tweets_matrix,zero_array),axis=1)




#Model Training
#..........................(1) RANDOM FOREST CLASSIFIER-----------------------------
from sklearn.ensemble import RandomForestClassifier
#classifier1 = RandomForestClassifier(n_estimators=200, random_state=0)
#model_building(classifier1,X_tweets,Y_train_tweets,new_test_tweets_matrix,Y_test_tweets,"RANDOM-FOREST-CLASSIFIER")#34.0%

#----------------------------(2) MULTINOMIAL NAIVE BAYES -----------------------
from sklearn.naive_bayes import MultinomialNB
#classifier2=MultinomialNB()
#model_building(classifier2,X_tweets,Y_train_tweets,new_test_tweets_matrix,Y_test_tweets,"MULTINOMIAL-NAIVE-BAYES-CLASSIFIER")#42%

#-----------------------------(3) SUPPORT VECTOR MACHINE(RBF)---------------------
from sklearn.svm import SVC
#classifier3=SVC(kernel='rbf')
#model_building(classifier3,X_tweets,Y_train_tweets,new_test_tweets_matrix,Y_test_tweets,"SUPPORT-VECTOR-MACHINE(RBF)-CLASSIFIER")#43.73%

#------------------------------(4) SUPPORT VECTOR MACHINE(LINEAR)---------------------------
from sklearn.svm import SVC
#classifier4=SVC(kernel='linear')
#model_building(classifier4,X_tweets,Y_train_tweets,new_test_tweets_matrix,Y_test_tweets,"SUPPORT-VECTOR-MACHINE(LINEAR)-CLASSIFIER")#41%

#---------------------------------(5) XGBOOST CLASSIFIER -----------------------------
from xgboost import XGBClassifier
classifier6=XGBClassifier()
model_building(classifier6,X_tweets,Y_train_tweets,new_test_tweets_matrix,Y_test_tweets,"XGBOOST-CLASSIFIER")#43.43%

#-----------------------------------(6) LOGISTIC REGRESSION -----------------------------------
from sklearn.linear_model import LogisticRegression
#classifier7=LogisticRegression()
#model_building(classifier7,X_tweets,Y_train_tweets,new_test_tweets_matrix,Y_test_tweets,"LOGISTIC-REGRESSION")#41.0%

















