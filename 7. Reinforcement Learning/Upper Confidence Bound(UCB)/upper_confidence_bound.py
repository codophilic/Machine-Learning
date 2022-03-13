# Upper Confidence Bound (UCB)

#Here we finding which ads design will be better for Marketing.
#There are 10 version of design and 0,1 are the like or dislike by the users.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB(MATH OF UCB)
import math
N = 10000 #Total number of users.
d = 10 #10 versions.
ads_selected = [] #in each round it appends i version of ad and keeps track which ad has higher UCB.
#Step1:
numbers_of_selections = [0] * d #(Ni)No. of times ad 'i'(1-10) was selected in 10000 rounds. 
sums_of_rewards = [0] * d #(Ri)sum of rewards of the ad 'i' up to 10000 rounds  
total_reward = 0#maximize total reward.

#Step2:
for n in range(0, N): #for loop of users
    ad = 0
    max_upper_bound = 0
    for i in range(0, d): #for loop of 10 version
        if (numbers_of_selections[i] > 0):#--> In first round all number of selection is 0.
            #beacuse all list values are 0 so in the formula it will give error.
            #And also helps to set same values to all 10 version UCB in 'else loop'. 
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            #n=0 therefore n start from 1 for log 
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400#(10^4
            #set initial UCB large value
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound #max UCB of these 10 versions.
            ad = i#-->(1-10)
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    #tells us which ad was selected most-->ad no. 4 
    
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    #tells us which ad has received highest reward---> ad no. 4 here
    
    total_reward = total_reward + reward

# Visualising the results
#Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()