#reinforcement learning
#a very powerful tool to predict the action at timt t+1 when the data till t is known
#it is also used for training AI

#creating a robot dog, making him learn how to move 

#multi armed bandit problem

#basically we have 5 or 10 different ad's or objects from which we have to choose the most optimal one
#we use Upper Confidence Bound 
#we can also do A/B testing to compare them and choose the best but it is only an exploration technique 
#while exploring, it can incur us a lot of cost 
#UCB has an advantage of exploring the best option and at the same time exploiting it to get the benefit out of it

#intution
#from the distribution curve we take the value of on x axis and plot on the y-axis of another graph
#we make an initial guess for the values on y-axis and draw a confidence bound around them 
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)')

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
#defining 2 variables
#first one is number of times an add i is selected upto n rounds 
#we have to consider this number for all the ads, so we'll create a vector of size d[number of ads]


#since we have to comupte for each round n, we will run a for loop of n from 1 to 10000 (total number of rounds or users)
N=10000
d=10

number_of_selections = [0]*d
#second variable is the sum of rewards of an ad i upto round n
sums_of_rewards = [0]*d

#total reward
total_reward = 0

#we create a list of the different versions of ads that were selected at each round (a huge vector)
ads_selected = []

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(number_of_selections[i]>0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward+delta_i
        else:
            upper_bound=1e400;
        if upper_bound>max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad]=sums_of_rewards[ad]+reward
    total_reward = total_reward + reward
    
#visualizing the result
plt.hist(ads_selected)
plt.title('Histogram of Selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')