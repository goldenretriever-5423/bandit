import pandas as pd
import os

os.chdir('/media/yuting/TOSHIBA EXT/retweet/data_processing/')
header=['idretweet' , '0', '1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9','10','11','12','13', 'date' ,'regular_node' ,'influencer' ,'original_tweet']

#tweets = pd.read_csv("5_date_sorted_influencer_10encoded_retweets.csv", delimiter=";")
#tweets = pd.read_csv("4_influencer_encoded_retweets.csv", delimiter=';')
tweets = pd.read_csv('4_influencer_10encoded_retweets.csv', index_col=False, delimiter=';', header=None, names=header)

tweets['context'] = tweets['0'].astype(str)
for i in range(1,14):
    tweets['context'] += ' ' + tweets[str(i)].astype(str)

# keep only the relevant columns
kept_tweets = tweets[['date','regular_node','influencer','original_tweet','context']]

grouped = kept_tweets.groupby(['influencer', 'context'])

# count the regular nodes per (influencer, context)
data = pd.merge(kept_tweets, grouped.agg({'regular_node': 'count'}), how='right', on=['influencer', 'context'], suffixes=('','_count')) 
# count unique regular nodes per (influencer, original_tweet)
data = pd.merge(data, grouped.agg({'regular_node': 'unique'}), how='right', on=['influencer', 'context'], suffixes=('','_set_unique')) 
data['unique_activations'] = data['regular_node_set_unique'].str.len()

# duplicates of (influencer, context) are irrelevant, because the number of activations is aggregated 
data = data.drop_duplicates(['influencer', 'context'])


data.date = pd.to_datetime(data.date, errors='coerce')
data.dropna(inplace=True)
data.sort_values(by=['date'], inplace=True)



overall_unique_activations = set()
influencers = [130734452, 95023423, 972651, 26257166, 125786481, 14511951, 233430873, 807095, 126400767, 813286]
selections = dict.fromkeys(influencers, 0)
total_reward = dict.fromkeys(influencers, 0)

data['selections'] = 0
data['new_activations'] = 0

for row in data.itertuples():
    if row.influencer not in influencers:
        print(row.influencer)
        continue
    selections[row.influencer] += 1
    data.loc[row.Index, 'selections'] = selections[row.influencer]
    data.loc[row.Index, 'new_activations'] = len(set(row.regular_node_set_unique.flatten()).difference(overall_unique_activations))
    overall_unique_activations = overall_unique_activations.union(set(row.regular_node_set_unique.flatten()))

# messy entries
#data.drop('regular_node', axis=1, inplace=True)
# data.drop('regular_node_set_unique', axis=1, inplace=True)

data.to_csv("6_date_sorted_influencer_10context_data.csv", sep=";", index=False)
