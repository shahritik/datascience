setwd('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)')

#import the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

#we will solve the multi armed bandit problem 
#selecting a random version of ad and check the total reward
#the total reward is 1242

#implementing UCB
N=10000
d=10
total_reward=0
numbers_of_selection = integer(d)
sums_of_rewards = integer(d)
ads_selected = integer(0)
for (n in 1:N){
  ad=0
  max_upperbound = 0
  for(i in 1:d)
  {
    if(numbers_of_selection[i]>0)
    {
    average_reward = sums_of_rewards[i]/numbers_of_selection[i]
    delta_i = sqrt(3/2*log(n)/numbers_of_selection[i])
    upper_bound = average_reward+delta_i
    }else{
      upper_bound = 1e400
    }
    if(upper_bound>max_upperbound){
      max_upperbound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selection[ad] = numbers_of_selection[ad]+1
  reward = dataset[n,ad]
  sums_of_rewards[ad]=sums_of_rewards[ad]+reward
  total_reward = total_reward + reward
}

hist(ads_selected)
