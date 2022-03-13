# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#The header is simply names of the product and not title of the column transaction products. 
#Thats we remove it.
#Total No. of transaction=7500 
#Total product=20

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
#The module needs list in list-->[[]] and not a dataFrame.
#E.g [['A','B','C'..],['D','E','F'..],..]


# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#min_length-->sets minimum numbers of product in rules. 
#Suppose a product is purchase 3 times a day therefore in a week its purchase 21 times
#min_support=21/7500=0.003(approx)

#If min_confidence is set too high e.g 0.8 it means that the rules has tooo be correct
#80% of the cases or 4 out of 5. So we get some rules containing some products that are most
#purhcase but they are not like logically associated or not for the right reason they can be put in
#Same basket. E.g people in summer buy mineral water and some people like omlet and they purchase eggs
#So the people buy lots of eggs and water we cannot logically associate them in same stall in the store.
#No proper reason or not relevant. So if we set confidence too high we can ended up such products
#in our learning rules.

#So a min_support=0.03 and min_confidence=0.2 are combination of good rules.
#You can try different combinations for these choices.  

# Interpretation of results
results = list(rules)
print(results)