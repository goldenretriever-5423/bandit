import numpy as np
from timeit import default_timer as timer
import pdb
import math
import random as rand
import networkx as nx
import operator
import os
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.spatial import distance
from operator import add, itemgetter
from networkx import barabasi_albert_graph
from scipy.stats import wasserstein_distance
from sklearn import preprocessing
import seaborn as sns

# from numerize import numerize

import numpy as np


from joblib import Parallel, delayed

import warnings


def get_reward(context, influencer_idx):
    influencer = INFLUENCERS[influencer_idx]
    rewards = tweets.loc[tweets.context == ' '.join(context[:DCONTEXT + 1].astype(str)), :].loc[
        tweets.influencer == influencer]
    return 0 if rewards.empty else rewards.new_activations.iloc[0]


def get_all_activations(context, influencer_idx):
    influencer = INFLUENCERS[influencer_idx]
    rewards = tweets.loc[tweets.context == ' '.join(context[:DCONTEXT + 1].astype(str)), :].loc[
        tweets.influencer == influencer]
    return 0 if rewards.empty else rewards.regular_node_count.iloc[0]


def get_seed_rewards(contexts):
    all_rewards = [[get_reward(context, influencer_idx) for influencer_idx in np.arange(K)] for context in contexts]
    return all_rewards


def get_tweet(context, influencer_idx):
    influencer = INFLUENCERS[influencer_idx]
    tweet = tweets.loc[tweets.context == ' '.join(context[:DCONTEXT + 1].astype(str)), :].loc[
        tweets.influencer == influencer]
    if tweet.size > 0:
        tweet = tweet.sample()
    return tweet


def get_seed_tweets(contexts): # taking too much time, why can't just use direct mapping?
    campaign = []
    for context in contexts:
        tweets_temp = pd.DataFrame()
        for influencer_idx in np.arange(K):
            tweets_temp = pd.concat([tweets_temp, get_tweet(context, influencer_idx)])
        campaign.append(tweets_temp)
    return campaign


##################################################################
# LinUCB
##################################################################
def linucb_reward_and_selections(seed, campaign, contexts, lognorm=False):

    BUDGET = H

    # 1-delta confidence interval
    exploration_factor = np.sqrt(np.log(2 * BUDGET * K / delta) / 2)   # alpha

    influencer_hist_lin = []
    activations_hist_lin_all = []

    for tt in range(T):
        theta_estimators = np.empty(shape=(BUDGET, K, DCONTEXT))  # for each round, for each influencer
        reward_estimators = np.empty(shape=(BUDGET, K))  # for each round, for each influencer

        # selections per influencer
        selections_hist = dict.fromkeys(np.arange(K), 0)

        V = [np.diag(np.ones(DCONTEXT)) for _ in np.arange(K)] # A identity matrix
        observed_reward = [np.zeros(DCONTEXT) for _ in np.arange(K)] # b d=24 context length

        prev_activated = set()

        # initialization phase (with separate contexts)
        for k in np.arange(K):
            context = contexts[tt * H + k] # N*24?
            context = np.nan_to_num(context)
            context = preprocessing.normalize(context[:, np.newaxis], axis=0, norm='l2').ravel()

            # k-> tt
            tweet = campaign[tt][campaign[tt].influencer == INFLUENCERS[k]]
            acts = set()
            if not tweet.empty:
                acts = tweet.regular_node_set_unique.values[0]
            reward = len(acts - prev_activated)
            prev_activated.update(acts)

            activations_hist_lin_all.append(reward)
            influencer_hist_lin.append(k)
            training_reward = np.log(reward + 1) / MAX_LOG_NA if lognorm else reward / MAX_NA ## a scaling process, map # of nodes activated into [0,1]
            V[k] += np.matmul(context[np.newaxis].T, context[np.newaxis])
            observed_reward[k] += training_reward * context
            selections_hist[k] += 1

        # def f(x):
        #     return -x / BUDGET

        for t in np.arange(K, BUDGET):
            context = contexts[tt * H + t]
            context = np.nan_to_num(context)
            context = preprocessing.normalize(context[:, np.newaxis], axis=0, norm='l2').ravel()
            for k in np.arange(K):
                inv_V = np.linalg.inv(V[k])
                theta_estimators[t, k] = inv_V.dot(observed_reward[k])
                theta_estimators[t, k] = np.nan_to_num(theta_estimators[t, k])
                theta_estimators[t, k] = preprocessing.normalize(theta_estimators[t, k][:, np.newaxis], axis=0,
                                                                 norm='l2').ravel()
                reward_estimators[t, k] = theta_estimators[t, k].dot(context) + exploration_factor * np.sqrt(
                    context.dot(inv_V).dot(context))

            # chosen influencer
            all_activations = set()
            best_ks = list()
            for _ in np.arange(L):
                # chosen influencer
                best_k = np.argmax(reward_estimators[t])
                best_ks.append(best_k)
                reward_estimators[t][best_k] = 0
                influencer_hist_lin.append(best_k)
                # t -> tt
                tweet = campaign[tt * H + t][campaign[tt * H + t].influencer == INFLUENCERS[best_k]]
                acts = set()
                if not tweet.empty:
                    acts = tweet.regular_node_set_unique.values[0]
                all_activations.update(acts)

            reward = len(all_activations - prev_activated)
            prev_activated.update(all_activations)

            activations_hist_lin_all.append(reward)

            training_reward = np.log(reward / L + 1) / MAX_LOG_NA if lognorm else reward / L / MAX_NA
            for best_k in best_ks:
                V[best_k] += np.matmul(context[np.newaxis].T, context[np.newaxis])
                observed_reward[best_k] += training_reward * context
                selections_hist[best_k] += 1

    return activations_hist_lin_all, influencer_hist_lin



if __name__ == '__main__':
    os.chdir('/media/yuting/TOSHIBA EXT/retweet/data_processing/')

    warnings.filterwarnings("ignore")

    NUM_CORES = 25
    NRUNS = 50  # T

    BUDGET = 3000
    DCONTEXT = 14  # length of context vector
    delta = 0.01  # TODO to be discussed

    np.random.seed(100)

    INFLUENCERS = [130734452, 95023423, 972651, 26257166, 125786481, 14511951, 233430873, 807095, 126400767, 813286]
    K = len(INFLUENCERS)  # # of total arms to be selected

    # date;regular_node_id;influencer;original_tweet;context;regular_node_count;unique_activations;selections;new_activations
    # tweets = pd.read_csv("data_processing/6_date_sorted_influencer_context_data_acts_set_24.csv", delimiter=";")
    tweets = pd.read_csv("6_date_sorted_influencer_10context_data.csv", delimiter=";")
    MAX_NA = float(tweets.new_activations.max())  # maximum new acticated nodes
    MAX_LOG_NA = np.log(tweets.new_activations.max())  # TODO to be dicussed

    tweets.regular_node_set_unique = tweets.regular_node_set_unique.apply(
        lambda txt: set([int(i) for i in txt.strip("'[").strip("]'").split() if i.isdigit()]))

    # all the contexts
    twitter_contexts = tweets.context
    # needed as vector for the regressions
    twitter_contexts = list(
        map(lambda context: np.array(context.split(), dtype=float), twitter_contexts))  # list of context array

    used_contexts = set()

    np.random.seed(100)
    seeds = dict.fromkeys(
        list(set([np.random.randint(1000) for _ in np.arange(NRUNS + 10)]))[:NRUNS])  # set the random seeds
    for seed in seeds.keys():
        np.random.seed(seed)
        context_idx = list(set([np.random.randint(0, len(twitter_contexts)) for _ in np.arange(BUDGET + 100)]))[:BUDGET]
        seeds[seed] = [twitter_contexts[idx] for idx in context_idx]  # map to corresponding seed, to build id

    H1 = []
    H2 = []
    for l in [1]: # select one influencer for each round [1,2,5]

        T = 50
        H = 30
        L = l
        # BUDGET = T * H

        for seed in seeds:
            np.random.seed(seed)
            contexts = seeds[seed]
            seed_tweets = get_seed_tweets(contexts)
            h1, h2 = linucb_reward_and_selections(seeds, seed_tweets, contexts)
            H1.append(h1)
            H2.append(h2)