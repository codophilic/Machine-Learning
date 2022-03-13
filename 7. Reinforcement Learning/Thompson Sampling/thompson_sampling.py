# Thompson Sampling

#10 version of ads select the best version of ad for Marketing.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000#total users.
d = 10#10 version of ads.
ads_selected = []
#Step1:
numbers_of_rewards_1 = [0] * d#--->Ni1(n)
numbers_of_rewards_0 = [0] * d#--->Ni0(n)

total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    #picking random ssamolees and finding the highest reward. 
    for i in range(0, d):
        #Step2:
        #Bayesian inference.    
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        #betavariate--->it will gives us some random draws of the beta distribution
        #of parameters that we choose.
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()