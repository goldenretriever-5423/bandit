#!/bin/bash

source_file="../retweetstext.csv" 
delimiter="#1#8#3#"
# now select the tweets from top influencers
# see 2_get_top_influencers.sh and 2_influencers.out

awk -F ${delimiter} 'BEGIN{ split("130734452,95023423,972651,26257166,125786481,14511951,233430873,807095,126400767,813286", influencers, ","); for(i in influencers) dict[influencers[i]]="" } ($5 in dict) {print}' ${source_file} > 3_influencer_retweets.csv
