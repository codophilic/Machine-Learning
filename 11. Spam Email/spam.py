import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#Here 1 dataset is used for training and prediction.
# Importing the dataset
dataset = pd.read_csv('spam_modified.csv')


#cleaning of text
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer#lemmitization.
clean_texts=[]
clean_stems=[]
clean_lem=[]
for i in range(0,5191):
    clean=re.sub(pattern='a-zA-Z', repl=' ', string=dataset['text'][i])
    clean=clean.lower()
    clean=clean.split()
    
    stemming=PorterStemmer()
    clean=[stemming.stem(word) for word in clean if not word in set(stopwords.words('english'))]
    clean=' '.join(clean)
    clean_stems.append(clean)
    
    lemmatizer=WordNetLemmatizer()
    clean=[lemmatizer.lemmatize(word) for word in clean]
    clean=''.join(clean)
    
    clean_lem.append(clean)
    clean_texts.append(clean)


    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(clean_texts).toarray()
Y=dataset.iloc[:,-1].values
vocabulary=cv.vocabulary_

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size = 0.2, random_state = 0)

#from xgboost import XGBClassifier #97.58%
#classifier = XGBClassifier()

#from sklearn.linear_model import LogisticRegression #GaussianNB()-->52%
#classifier = LogisticRegression() #97.30%

from sklearn.naive_bayes import MultinomialNB 
#For text classification use MultinomialNB
classifier = MultinomialNB() #97.40%

#from sklearn.svm import SVC
#classifier=SVC(C=100,gamma=0.001,kernel='rbf')


classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

#-----------------------------------------------------------------------------
#Applying Manual Input, below mail sentences are spam.
text_input=['Registered Member Profile Shortlisted To Apply For Credit Card from State Bank Of India (SBI)'
            ,'Congratulations! Affordable Insurance for Coronavirus | Upto INR 1 CR Health Coverage Click here for details'
            ,'[Final Application Deadline Apr2] GET RECRUITED BY CARTESIAN, TESCO, INDEGENE | BUILD A REWARDING CAREER IN DATA SCIENCE'
            ]

clean_input_text=[]
#Applying same procedures/ data cleaning 
for i in range(len(text_input)):
    text_input_clean=re.sub(pattern='a-zA-Z', repl=' ', string=text_input[i])
    text_input_clean=text_input_clean.lower()
    text_input_clean=text_input_clean.split()
    text_input_clean=[stemming.stem(word) for word in text_input_clean if not word in set(stopwords.words('english'))]
    text_input_clean=' '.join(text_input_clean)
    text_input_clean=[lemmatizer.lemmatize(word) for word in text_input_clean]
    text_input_clean=''.join(text_input_clean)
    clean_input_text.append(text_input_clean)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
tfidf=TfidfVectorizer("english")
X_manual_input=tfidf.fit_transform(clean_input_text).toarray()

#Here we have fitted a 2D array of (5191,42623) on the classifier.
#While prediciting the model we have a 2D array of (3,36) so thats gives error
#since column number are not match (36 & 42623). So we create additional column
#of zeros and concatenate them to the dataset.(42623-36 these many columns are need to be added)
 
zero_arrays=np.zeros([3,42587])
new_input=np.concatenate((X_manual_input,zero_arrays),axis=1)

spam_or_ham=classifier.predict(new_input)
spam_or_ham=pd.DataFrame(spam_or_ham)
spam_or_ham=spam_or_ham.replace(to_replace=[0,1],value=['ham','spam'])
#-----------------------------------------------------------------------------


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)


Accuracy=classifier.score(X_test, Y_test)
print("Accuracy",round(Accuracy,4)*100,"%")



