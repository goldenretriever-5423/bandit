import os
import numpy as np
from gensim import downloader
from gensim.summarization.textcleaner import clean_text_by_word
from scipy import spatial


delimiter = "#1#8#3#"
source_file = "3_influencer_retweets.csv"

# obtain with get_top_influencers.sh
influencers = ['130734452', '95023423', '972651', '26257166', '125786481', '14511951', '233430873', '807095', '126400767', '813286']

#context_vectors_file = "1_10context_vectors.out"
#context_vectors_file = "1_context_vectors_8subclustered.out"
context_vectors_file = '1_subclustered_10context_vectors.out'
file_out = "4_influencer_encoded_retweets_8subclustered14D.csv"


class DatasetCreator(object):

    def __init__(self, source_file):
        self.delimiter = delimiter

        self.file = source_file
        self.file_out = file_out

        self.context_classes = list()
        with open(context_vectors_file,  'r') as f:
            lines=f.readlines()
            [self.context_classes.append([float(dim) for dim in line.split()]) for line in lines]
        self.contextTree=spatial.KDTree(self.context_classes)

        self.model = downloader.load("glove-twitter-200")


    def create_dataset(self):
        with open(self.file_out, "w") as fout:
            with open(self.file, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    # 0 idretweet #1#8#3# 
                    # 1 content of message #1#8#3#
                    # 2 date of action RT #1#8#3#
                    # 3 iduser having retweeted the tweet #1#8#3#
                    # 4 iduser of original tweet #1#8#3#
                    # 5 id of original tweet
                    retweet = str(line).split(self.delimiter)
                    if( len(retweet) < 6):
                        print(line)
                        continue

                    original_author = retweet[4]
                    text = retweet[1].replace("\n", " ")

                    words= clean_text_by_word(text).keys()
                    centroids = [0] * len(self.context_classes)
                    count = 0
                    for word in words:
                        try:
                            centroids[self.contextTree.query(self.model.word_vec(word))[1]] += 1
                            count += 1
                        except:
                            pass
                    centroids = [str(float(c)/count) if count != 0 else str(0) for c in centroids]
                    encoded = ";".join(centroids)
                    fout.write(";".join([retweet[0], encoded, retweet[2], retweet[3], retweet[4], retweet[5]]))

DatasetCreator(source_file).create_dataset()
# add header: retweet_id;0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;date;regular_node;influencer;original_tweet
