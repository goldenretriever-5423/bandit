import sys
import pandas as pd


delimiter = ";"
source_file = "4_influencer_10encoded_retweets.csv"
date_index = 11

df = pd.read_csv(source_file, delimiter=delimiter, header=None)
df[date_index] = pd.to_datetime(df[date_index])
df.sort_values(by=[date_index], inplace=True)
df.to_csv("5_date_sorted_influencer_10encoded_retweets.csv", sep=delimiter, index=False, header=['idretweet' ,'1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'date' ,'regular_node' ,'influencer' ,'original_tweet'])
