#Association rule learning 
#person who bought also bought 
#checking the correlation between the products 
#if bread and milk are brought together they will be place far apart in the store so that people have to walk through the whole store
#they get time to see other products which may entice them 
#while some stores place this close to each other so that the products can come into the same basket
#how does the algorithm works: it has 3 parts support, confidence and the lift
#support: number of people who have watched a particular movie/total number of people
#confidence: number of people who have seen both movie m1 and m2/divided number of people who have seen m2 
#lift: lift is confidence/support 
#lift is basically imporvement in your prediction: if a person who already likes movie 2 and movie 1 is suggested to them then greater chances that they will like movie 1
#summary
#step-1: set up a minimum support and confidence 
#step-2: take all the subsets having higher support than than minimum support 
#step-3: take all the subsets having higher confidence than the minimum confidence 
#step-4: sort the rules by decreasing lift 
#one with highest lift is the strongest 
#recommender systems of ecommerce are based on the logic of apriori, but they are very advanced algorithms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os 

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori')
dataset=pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#optimizing the sales
#to preapare the dataset for apriori algorithm we prepare a list of list
transactions = [] 
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#training the apriori dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#visualizing the results
results = list(rules)