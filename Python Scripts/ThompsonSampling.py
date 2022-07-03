#multi arm bandit problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling')

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#we have to find the ideal difference between exploration and exploitation 
#the x axis shows the return

