# Natural Language Processing

#Restaurant reviews wheather it is positive or negative.
#predicting reviews were positive or negative
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#It contain all reviews of customer as a text
#positive review-->1
#negative review-->0.
#the delimiter separating two column in this .tsv file is tab.(Tab Separated Values-->tsv)
#Comma Separated Value-->CSV.
#We use tsv file because if we use csv file it would interpret reviews having comma as 
#new column. e.g Overall,the food was Nice.
#So in tsv file we have two columns 1) Review 2)0/1
#quoting-->ignore double quotes.

# Cleaning the texts(e.g The,that,these...etc)
#get rid of punctuations & numbers.
#In this we will also apply Stemming.
#Making all text lowercase. 
import re
import nltk
nltk.download('stopwords')#download the list of stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_= []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #This will keep only the small alphabets,capital alphabets and spaces.
    #removes all the numbers and punctuations,
    #e.g wow...i love the Place
    #output: wow   i love the Place
    #' '--> the remove characters are replace by spaces or else it will give error.
    review = review.lower()#convert string to lower case.
    review = review.split()
    ps = PorterStemmer()#stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #The stopwords contains word which are irrelavant 
    #e.g Wow   loved this place--> here 'this' will not help us to give predictions.
    #So it is irrelavant.
    #Here all reviews are in english so we use stopwords of english language.
    #using 'set' helps to make our algorithm run faster.
    review = ' '.join(review)
    corpus_.append(review)
    #If we dont apply Stemming there will huge list of words and hence it would 
    #makes algorithm sslower.

# Creating the Bag of Words model
#Tokenization.

#collects unique words and make a column.
#e.g text1-->wow love place
#    text2-->good
#    ....so on
#makes a column(bag of words)
#       wow love good 
#text1   1   1    0
#text2   0   0    1
#..
#it will make a column in which column-words are find.(1)
#it will be a sparse matrix(matrix having many 0)  andd we reduce the sparse matrix 
#as much as possible. 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
#(max_features = 1500)#also does tokenization. 
#max_feature helps to reduce words which are most repeated many times
X = cv.fit_transform(corpus_).toarray()
#It tokenize and build vocab with key-value pairs
#corpus_ is a list so to convert into array we use '.toarray()'.
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#We use classification mmodel since we are classifying positive reviews and negative reviews.
# Training the Naive Bayes model on the Training set
#naive_bayes are commonly used.
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)