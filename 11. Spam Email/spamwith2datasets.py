import pandas as pd

# Importing the dataset
dataset = pd.read_csv('spam_modified.csv')
dataset1 = pd.read_csv('spam2.csv')
dataset1=dataset1.replace(to_replace=['ham','spam'],value=[0,1])
#cleaning of text
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer#lemmitization.

clean_texts1=[]
for i in range(0,5191):
    clean=re.sub(pattern='a-zA-Z', repl=' ', string=dataset['text'][i])
    clean=clean.lower()
    clean=clean.split()
    
    stemming=PorterStemmer()
    clean=[stemming.stem(word) for word in clean if not word in set(stopwords.words('english'))]
    clean=' '.join(clean)
    
    
    lemmatizer=WordNetLemmatizer()
    clean=[lemmatizer.lemmatize(word) for word in clean]
    clean=''.join(clean)
    
    
    clean_texts1.append(clean)#corpus1

clean_texts2=[]
for i in range(0,5572):
    clean=re.sub(pattern='a-zA-Z', repl=' ', string=dataset1['v2'][i])
    clean=clean.lower()
    clean=clean.split()
    
    stemming=PorterStemmer()
    clean=[stemming.stem(word) for word in clean if not word in set(stopwords.words('english'))]
    clean=' '.join(clean)
    
    
    lemmatizer=WordNetLemmatizer()
    clean=[lemmatizer.lemmatize(word) for word in clean]
    clean=''.join(clean)
    
    
    clean_texts2.append(clean)#corpus2

    
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer("english")
X1=tfidf.fit_transform(clean_texts1).toarray()
Y1=dataset.iloc[:,-1].values

X2=tfidf.transform(clean_texts2).toarray()
#Say you have a corpus made of 33 different words, then your bag of words 
#at training time will have 33 columns. Now you are using another corpus 
#which has only 4 different words. You end up with a matrix with 4 columns, 
#and the model won't like that! hence you need to fit the second corpus in 
#the same bag of words matrix you had at the beginning, with 33 columns.
#Solution-->For example one way is to save the transform object you used at 
#training time with fit() and then apply it at test time (only transform())!
#So fit_transform 1 dataset and just tranform another dataset.
Y2=dataset1.iloc[:,0].values

X_train,Y_train=X1,Y1
X_test,Y_test=X2,Y2

#If we use count vectorization and GaussianNB we get 52.4% accuracy.

from sklearn.naive_bayes import MultinomialNB 
#For text classification use MultinomialNB
classifier = MultinomialNB()

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)


Accuracy=classifier.score(X_test, Y_test)
print("Accuracy",round(Accuracy,3)*100,"%")



