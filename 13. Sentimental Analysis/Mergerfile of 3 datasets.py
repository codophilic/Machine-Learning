#While combining the two datasets i.e tweet_3-point & tweet1_3-point datasets and predicting the test.csv dataset
#gives less accuracy so thats the reason merging all the 3 datasets.

import pandas as pd

df1=pd.read_csv('tweet_3-point.csv')
df2=pd.read_table('tweet1_3-point.tsv')
df3=pd.read_table('test.tsv')

new_dataset_text=pd.concat([df1['text'],df2['tweet'],df3['tweet']],ignore_index=True)


new_dataset_senti=pd.concat([df1['airline_sentiment'],df2['label'],df3['label']],ignore_index=True)
#print(new_dataset_senti)

new_dataset=pd.DataFrame({
    'text':new_dataset_text,
    'Sentiments':new_dataset_senti
})
new_dataset['Sentiments']=new_dataset['Sentiments'].replace(['neutral','positive','negative',-1,1,0],[0,1,2,0,1,2])
#0=Neutral,1=Positive,2=Negative
print(new_dataset.shape)
print(new_dataset)


new_dataset.to_csv('Merge_sentiment_dataset.csv',index=False)


