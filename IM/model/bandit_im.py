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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, newton
from scipy.special import factorial
from scipy.stats import poisson
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed

import warnings

warnings.filterwarnings("ignore")

NUM_CORES = 25
NRUNS = 50

BUDGET = 3000
K = 10
DCONTEXT = 24
delta = 0.01

FATGTUCB = 'FAT-GT-UCB'
RLFATGTUCB = 'RL-FAT-GT-UCB'
RLLSVIGTUCB = 'LSVI-GT-UCB'
LSVIUCB = 'LSVI-UCB'
LSVIUCBST = 'LSVI-UCB - separate thetas'

np.random.seed(100)

INFLUENCERS = [130734452, 95023423, 972651, 26257166, 125786481, 14511951, 233430873, 807095, 126400767, 813286]
K = len(INFLUENCERS)

# date;regular_node_id;influencer;original_tweet;context;regular_node_count;unique_activations;selections;new_activations
# tweets = pd.read_csv("data_processing/6_date_sorted_influencer_context_data_acts_set_24.csv", delimiter=";")
# MAX_NA = float(tweets.new_activations.max())
# MAX_LOG_NA = np.log(tweets.new_activations.max())

MAX_NA = 100.0
MAX_LOG_NA = 2.0
#
# tweets.regular_node_set_unique = tweets.regular_node_set_unique.apply(
#     lambda txt: set([int(i) for i in txt.strip("'[").strip("]'").split() if i.isdigit()]))
#
# # all the contexts
# twitter_contexts = tweets.context
# # needed as vector for the regressions
# twitter_contexts = list(map(lambda context: np.array(context.split(), dtype=float), twitter_contexts)) # list of context array
#
# used_contexts = set()
#
# np.random.seed(100)
# seeds = dict.fromkeys(list(set([np.random.randint(1000) for _ in np.arange(NRUNS + 10)]))[:NRUNS])
# for seed in seeds.keys():
#     np.random.seed(seed)
#     context_idx = list(set([np.random.randint(0, len(twitter_contexts)) for _ in np.arange(BUDGET + 100)]))[:BUDGET]
#     seeds[seed] = [twitter_contexts[idx] for idx in context_idx] # map to corresponding seed, to build id


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


def get_seed_tweets(contexts):
    campaign = []
    for context in contexts:
        tweets = pd.DataFrame()
        for influencer_idx in np.arange(K):
            tweets = pd.concat([tweets, get_tweet(context, influencer_idx)])
        campaign.append(tweets)
    return campaign


##################################################################
# LinUCB
##################################################################
def linucb_reward_and_selections(seed, campaign, contexts, lognorm=False):
    T = 50
    H = 30
    L = 2
    BUDGET = T * H
    H = 30
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
            training_reward = np.log(reward + 1) / MAX_LOG_NA if lognorm else reward / MAX_NA
            V[k] += np.matmul(context[np.newaxis].T, context[np.newaxis])
            observed_reward[k] += training_reward * context
            selections_hist[k] += 1

        def f(x):
            return -x / BUDGET

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
    seeds = []
    context = []
    campain = []
    h1, h2 = linucb_reward_and_selections(seeds, campain, context)



'''
##################################################################
# FAT-GT-UCB
##################################################################
def fatgt_reward_and_selections(seed, campaign, contexts):
    st = timer()
    influencer_hist_fatgt = []
    activations_hist_fatgt_all = []

    for tt in range(T):
        prev_activated = set()

        # rewards history for each influencer
        activations_hist_fatgt = dict.fromkeys(np.arange(K))
        for k in np.arange(K):
            activations_hist_fatgt[k] = []

        # selections per influencer
        selections_hist = dict.fromkeys(np.arange(K), 0)

        # initialization phase
        # play each influencer once
        for k in np.arange(K):
            # observe the spread (set of activated basic nodes)
            tweet = campaign[tt * H + k][campaign[tt * H + k].influencer == INFLUENCERS[k]]
            acts = set()
            all_activations = 0
            if not tweet.empty:
                acts = tweet.regular_node_set_unique.values[0]
                all_activations = tweet.regular_node_count.values[0]
            reward = len(acts - prev_activated)
            prev_activated.update(acts)

            activations_hist_fatgt_all.append(reward)
            activations_hist_fatgt[k].append(reward)
            influencer_hist_fatgt.append(INFLUENCERS[k])
            # GT stats
            selections_hist[k] += 1

        # UCB algorithm
        for t in np.arange(K, H):
            best_k_ucbs = list()
            best_k_contexts = list()
            for k in np.arange(K):
                # The UCB of the remaining potential
                if selections_hist[k] == 0:
                    gt = beta = 0
                else:
                    gt = 1 / float(selections_hist[k] ** 2) * np.sum(
                        [activations_hist_fatgt[k][s] * s for s in np.arange(selections_hist[k])])
                    beta = (1 + np.sqrt(2)) * np.sqrt(
                        np.sum([activations_hist_fatgt[k][s] * s for s in np.arange(selections_hist[k])]) * np.log(
                            4 * t) / float(selections_hist[k] ** 2 * (selections_hist[k] + 1))) + np.log(4 * t) / float(
                        selections_hist[k]) / 3
                b_k = gt + beta
                best_k_ucbs.append(b_k)

            bks = list()
            for l in np.arange(L):
                max_ucb = -1
                best_k = 0
                for k, ucb in enumerate(best_k_ucbs):
                    if max_ucb < ucb and k not in bks:
                        max_ucb = ucb
                        best_k = k
                best_k_ucbs[best_k] = -1
                bks.append(best_k)

            all_acts = set()
            for l in np.arange(L):
                best_k = bks[l]
                influencer_hist_fatgt.append(INFLUENCERS[best_k])
                # observe the spread (set of activated basic nodes)
                tweet = campaign[tt * H + t][campaign[tt * H + t].influencer == INFLUENCERS[best_k]]
                acts = set()
                if not tweet.empty:
                    acts = tweet.regular_node_set_unique.values[0]
                all_acts.update(acts)

            reward = len(all_acts - prev_activated)
            prev_activated.update(all_acts)

            activations_hist_fatgt_all.append(reward)
            # GT stats
            for best_k in bks:
                activations_hist_fatgt[best_k].append(reward)
                selections_hist[best_k] += 1

    et = timer()
    elapsed_time = et - st
    return activations_hist_fatgt_all, influencer_hist_fatgt, elapsed_time


##################################################################
# Lognorm-LSVI-UCB
##################################################################
def lsvi_reward_and_selections(seed, campaign, contexts):
    st = timer()
    c = 0.0005  # an absolute constant c > 0
    d = DCONTEXT + K  # PHI_SIZE + 1
    paperT = T * H  # the total number of steps
    prob = 0.01  # with probability 1-p the total regret is at most
    exploration_factor = c * d * H * np.sqrt(np.log(2 * d * paperT / prob))

    influencer_hist = []
    activations_hist_all = []

    # rewards history for each influencer
    activations_hist = dict.fromkeys(np.arange(K))

    # the estimated Q-function
    Q = dict.fromkeys(np.arange(T))
    for t in np.arange(T):
        Q[t] = dict.fromkeys(np.arange(H))
        for h in np.arange(H):
            Q[t][h] = dict.fromkeys(np.arange(K), 0)

    # the estimator of the unknown parameter theta, the Q-function, and the state and action history
    actions_hist = dict.fromkeys(np.arange(-1, T + 1))
    theta_estimators = dict.fromkeys(np.arange(-1, T + 1))
    rewards_hist = dict.fromkeys(np.arange(-1, T + 1))
    for t in np.arange(-1, T + 1):
        actions_hist[t] = dict.fromkeys(np.arange(-1, H + 1))
        theta_estimators[t] = dict.fromkeys(np.arange(-1, H + 1))
        rewards_hist[t] = dict.fromkeys(np.arange(-1, H + 1), 0)
        for h in np.arange(-1, H + 1):
            actions_hist[t][h] = np.zeros(K)
            theta_estimators[t][h] = np.zeros(d)
    design_matrix = dict.fromkeys(np.arange(-1, H + 1))
    for h in np.arange(-1, H + 1):
        design_matrix[h] = np.identity(d)

    def phi(tau, h):
        context = contexts[tau * H + h]
        context = np.nan_to_num(context)
        context = preprocessing.normalize(context[:, np.newaxis], axis=0, norm='l2').ravel()
        action = actions_hist[tau][h - 1]
        state = np.array([*context, *action])
        state = preprocessing.normalize(state[:, np.newaxis], axis=0, norm='l2').ravel()
        return state

    def estimate_Qfunction(tau, t, h):
        f = phi(tau, h)
        inv_Sigma = np.linalg.inv(design_matrix[h])
        Qestimator = f.dot(theta_estimators[t][h]) + exploration_factor * np.sqrt(f.dot(inv_Sigma).dot(f))
        return Qestimator

    def take_action(tau, t, h):
        maxQ = dict.fromkeys(np.arange(K), 0)
        maxAction = np.zeros(K)

        if h >= H:
            return maxQ, maxAction

        l = 0
        actions = [i for i in product(range(2), repeat=K) if sum(i) == 1]
        actions.sort(reverse=True)
        while l != L:
            maxq = -1
            maxa = np.zeros(K)
            for action in actions:
                # if the action has already been selected
                if maxAction[action.index(1)] == 1:
                    continue
                # Qfunction and state at horizon h, state from episode tau, Qfunction estimate in episode t
                # \hat{Q}_{h,t}(s_{h,\tau},a) = \langle \phi(s_{h,\tau}, a), \hat{\theta}_{h,t} + exploration_factor * \sqrt{phi^T(s_{h,\tau},a) \Sigma^{-1}_{h,t} \phi(s_{h,\tau},a)}
                q = estimate_Qfunction(tau, t, h)
                if q > maxq:
                    maxq = q
                    maxa = action
            maxQ[maxa.index(1)] = maxq
            maxAction += maxa
            l += 1
        return maxQ, maxAction

    def get_training_rewards(t, h):
        training_rewards = list()
        for tau in np.arange(t):
            # \hat{V}_{h+1,t} = max_{a \in \mathcal{A}} \hat{Q}_{h+1,t}(s_{h+1, \tau}, a)
            maxV, maxAction = take_action(tau, t, h + 1)
            if lognorm:
                y = np.log(rewards_hist[tau][h]) / MAX_LOG_NA + sum(maxV)
            else:
                y = rewards_hist[tau][h] / MAX_NA + sum(maxV)
            training_rewards.append(y)
        return training_rewards

    def play_action(t, h):
        all_activations = set()
        # updating the next step's data to be used when building its state
        for best_k in [k for k, e in enumerate(actions_hist[t][h]) if e == 1]:
            best_k_context = contexts[t * H + h]
            # observe the spread (set of activated basic nodes)
            tweet = campaign[t * H + h][campaign[t * H + h].influencer == INFLUENCERS[best_k]]
            acts = set()
            if not tweet.empty:
                acts = tweet.regular_node_set_unique.values[0]
            all_activations.update(acts)
            influencer_hist.append(best_k)
            actions_hist[t][h][best_k] = 1

        f = phi(t, h)
        design_matrix[h] += np.matmul(f[np.newaxis].T, f[np.newaxis])
        reward = len(all_activations - prev_activated_nodes)
        # mark spread as previously activated for the next round
        prev_activated_nodes.update(all_activations)
        rewards_hist[t][h] = reward
        rewards_hist[t][h + 1] = reward
        activations_hist_all.append(reward)

    for t in np.arange(0, T):
        # each episode is a campaign, and it comes with new rewards
        prev_activated_nodes = set()
        for h in np.arange(H - 1, -1, -1):
            training_rewards = get_training_rewards(t, h)

            inv_Sigma = np.linalg.inv(design_matrix[h])
            x = list([phi(tau, h) * training_rewards[tau] for tau in np.arange(t)])
            x.append(np.zeros(d))
            theta_estimators[t][h] = inv_Sigma.dot(sum(x))
            theta_estimators[t][h] = np.nan_to_num(theta_estimators[t][h])
            theta_estimators[t][h] = preprocessing.normalize(theta_estimators[t][h][:, np.newaxis], axis=0,
                                                             norm='l2').ravel()

        for h in np.arange(0, H):
            # take action
            maxQ, actions_hist[t][h] = take_action(t, t, h)
            # observe reward
            play_action(t, h)

    et = timer()
    elapsed_time = et - st
    return activations_hist_all, influencer_hist, elapsed_time


##################################################################
# Lognorm-LSVI-UCB separate thetas
##################################################################
def lsvi_reward_and_selections_septhetas(seed, campaign, contexts, lognorm=False):
    st = timer()
    c = 0.0005  # an absolute constant c > 0
    d = DCONTEXT + 1 + 1  # PHI_SIZE + 1
    paperT = T * H  # the total number of steps
    prob = 0.01  # with probability 1-p the total regret is at most
    exploration_factor = c * d * H * np.sqrt(np.log(2 * d * paperT / prob))

    influencer_hist = []
    activations_hist_all = []

    # rewards history for each influencer
    activations_hist = dict.fromkeys(np.arange(K))

    initialized = False

    # the estimated Q-function
    Q = dict.fromkeys(np.arange(T))
    for t in np.arange(T):
        Q[t] = dict.fromkeys(np.arange(H))
        for h in np.arange(H):
            Q[t][h] = dict.fromkeys(np.arange(K), 0)

    # the estimator of the unknown parameter theta, the Q-function, and the state and action history
    actions_hist = dict.fromkeys(np.arange(-1, T + 1))
    theta_estimators = dict.fromkeys(np.arange(-1, T + 1))
    rewards_hist = dict.fromkeys(np.arange(-1, T + 1))
    selections_hist = dict.fromkeys(np.arange(-1, T + 1))
    for t in np.arange(-1, T + 1):
        actions_hist[t] = dict.fromkeys(np.arange(-1, H + 1))
        theta_estimators[t] = dict.fromkeys(np.arange(-1, H + 1))
        rewards_hist[t] = dict.fromkeys(np.arange(-1, H + 1), 0)
        selections_hist[t] = dict.fromkeys(np.arange(-1, H + 1))
        for h in np.arange(-1, H + 1):
            actions_hist[t][h] = np.zeros(K)
            selections_hist[t][h] = np.zeros(K)
            theta_estimators[t][h] = dict.fromkeys(np.arange(K))
            for k in np.arange(K):
                theta_estimators[t][h][k] = np.zeros(d)
    design_matrix = dict.fromkeys(np.arange(-1, H + 1))
    for h in np.arange(-1, H + 1):
        design_matrix[h] = dict.fromkeys(np.arange(K))
        for k in np.arange(K):
            design_matrix[h][k] = np.identity(d)

    def phi(tau, h, k):
        context = contexts[tau * H + h]
        context = np.nan_to_num(context)
        context = preprocessing.normalize(context[:, np.newaxis], axis=0, norm='l2').ravel()
        th = h - 1
        while actions_hist[tau][th][k] == 0 and th > 0:
            th -= 1
        reward = [rewards_hist[tau][th]]
        action = [actions_hist[tau][th][k]]
        state = np.array([*context, *reward, *action])
        state = preprocessing.normalize(state[:, np.newaxis], axis=0, norm='l2').ravel()
        return state

    def estimate_Qfunction(tau, t, h, k):
        f = phi(tau, h, k)
        inv_Sigma = np.linalg.inv(design_matrix[h][k])
        Qestimator = f.dot(theta_estimators[t][h][k]) + exploration_factor * np.sqrt(f.dot(inv_Sigma).dot(f))
        return Qestimator

    def take_action(tau, t, h):
        maxQ = dict.fromkeys(np.arange(K), 0)
        maxAction = np.zeros(K)

        if not initialized:
            # initialization: play each influencer once (one influencer per step)
            maxAction[h] = 1
            maxQ[h] = estimate_Qfunction(tau, t, h, h)
            return maxQ, maxAction

        if h >= H:
            return maxQ, maxAction

        l = 0
        actions = [i for i in product(range(2), repeat=K) if sum(i) == 1]
        actions.sort(reverse=True)
        while l != L:
            maxq = -1
            maxa = np.zeros(K)
            for action in actions:
                # if the action has already been selected
                if maxAction[action.index(1)] == 1:
                    continue
                # Qfunction and state at horizon h, state from episode tau, Qfunction estimate in episode t
                # \hat{Q}_{h,t}(s_{h,\tau},a) = \langle \phi(s_{h,\tau}, a), \hat{\theta}_{h,t} + exploration_factor * \sqrt{phi^T(s_{h,\tau},a) \Sigma^{-1}_{h,t} \phi(s_{h,\tau},a)}
                q = estimate_Qfunction(tau, t, h, action.index(1))
                if q > maxq:
                    maxq = q
                    maxa = action
            maxQ[maxa.index(1)] = maxq
            maxAction += maxa
            l += 1
        return maxQ, maxAction

    def get_training_rewards(t, h, k):
        training_rewards = list()
        for tau in np.arange(t):
            # \hat{V}_{h+1,t} = max_{a \in \mathcal{A}} \hat{Q}_{h+1,t}(s_{h+1, \tau}, a)
            maxV, maxAction = take_action(tau, t, h + 1)
            if lognorm:
                y = np.log(rewards_hist[tau][h]) / MAX_LOG_NA + maxV[k]
            else:
                y = rewards_hist[tau][h] / MAX_NA + maxV[k]
            training_rewards.append(y)
        return training_rewards

    def play_action(t, h):
        all_activations = set()
        # updating the next step's data to be used when building its state
        for best_k in [k for k, e in enumerate(actions_hist[t][h]) if e == 1]:
            best_k_context = contexts[t * H + h]
            # observe the spread (set of activated basic nodes)
            tweet = campaign[t * H + h][campaign[t * H + h].influencer == INFLUENCERS[best_k]]
            acts = set()
            if not tweet.empty:
                acts = tweet.regular_node_set_unique.values[0]
            all_activations.update(acts)
            influencer_hist.append(best_k)

            f = phi(t, h, k)
            design_matrix[h][best_k] += np.matmul(f[np.newaxis].T, f[np.newaxis])
            actions_hist[t][h][best_k] = 1

        reward = len(all_activations - prev_activated_nodes)
        for best_k in [k for k, e in enumerate(actions_hist[t][h]) if e == 1]:
            # updating the next step's data to be used when building its state
            selections_hist[t][h + 1][best_k] += reward
        # mark spread as previously activated for the next round
        prev_activated_nodes.update(all_activations)
        rewards_hist[t][h] = reward
        activations_hist_all.append(reward)

    for t in np.arange(0, T):
        # each episode is a campaign, and it comes with new rewards
        prev_activated_nodes = set()
        for h in np.arange(H - 1, -1, -1):
            for k in np.arange(K):
                training_rewards = get_training_rewards(t, h, k)

                inv_Sigma = np.linalg.inv(design_matrix[h][k])
                x = list(
                    [phi(tau, h, k) * training_rewards[tau] if actions_hist[tau][h][k] == 1 else np.zeros(d) for tau in
                     np.arange(t)])
                x.append(np.zeros(d))
                theta_estimators[t][h][k] = inv_Sigma.dot(sum(x))
                theta_estimators[t][h][k] = np.nan_to_num(theta_estimators[t][h][k])
                theta_estimators[t][h][k] = preprocessing.normalize(theta_estimators[t][h][k][:, np.newaxis], axis=0,
                                                                    norm='l2').ravel()

        # initialized = False
        if not initialized:
            # initialization: play each influencer once
            for h in np.arange(K):
                _, actions_hist[t][h] = take_action(t, t, h)
                play_action(t, h)
            initialized = True
            for h in np.arange(K, H):
                # take action
                maxQ, actions_hist[t][h] = take_action(t, t, h)
                # observe reward
                play_action(t, h)
        else:
            for h in np.arange(0, H):
                # take action
                maxQ, actions_hist[t][h] = take_action(t, t, h)
                # observe reward
                play_action(t, h)

    et = timer()
    elapsed_time = et - st
    return activations_hist_all, influencer_hist, elapsed_time


##################################################################
# RL-FAT-GT-UCB
##################################################################
def rlfatgt_reward_and_selections(seed, campaign, contexts):
    st = timer()
    influencer_hist_fatgt = []
    activations_hist_fatgt_all = []

    # the estimator of the unknown parameter theta, the Q-function, and the state and action history
    gts = dict.fromkeys(np.arange(-1, H + 1))
    betas = dict.fromkeys(np.arange(-1, H + 1))
    for h in np.arange(-1, H + 1):
        gts[h] = dict.fromkeys(np.arange(K))
        betas[h] = dict.fromkeys(np.arange(K))
        for k in np.arange(K):
            gts[h][k] = list()
            betas[h][k] = list()

    initialized = False

    for tt in range(T):
        prev_activated = set()

        startH = 0

        # rewards history for each influencer
        activations_hist_fatgt = dict.fromkeys(np.arange(K))
        for k in np.arange(K):
            activations_hist_fatgt[k] = []

        # selections per influencer
        selections_hist = dict.fromkeys(np.arange(K), 0)

        if not initialized:
            # initialization phase
            # play each influencer once#tau times
            for k in np.arange(K):
                # observe the spread (set of activated basic nodes)
                tweet = campaign[tt * H + k][campaign[tt * H + k].influencer == INFLUENCERS[k]]
                acts = set()
                all_activations = 0
                if not tweet.empty:
                    acts = tweet.regular_node_set_unique.values[0]
                    all_activations = tweet.regular_node_count.values[0]
                reward = len(acts - prev_activated)
                prev_activated.update(acts)

                activations_hist_fatgt[k].append(reward)
                activations_hist_fatgt_all.append(reward)
                influencer_hist_fatgt.append(INFLUENCERS[k])
                # GT stats
                selections_hist[k] += 1

            initialized = True
            startH = K

        # UCB algorithm
        for t in np.arange(startH, H):
            best_k_ucbs = list()
            best_k_contexts = list()
            trueT = t
            for k in np.arange(K):
                if selections_hist[k] == 0:
                    gt = beta = 0
                else:
                    gt = 1 / float(selections_hist[k] ** 2) * np.sum(
                        [activations_hist_fatgt[k][s] * s for s in np.arange(selections_hist[k])])
                    beta = (1 + np.sqrt(2)) * np.sqrt(
                        np.sum([activations_hist_fatgt[k][s] * s for s in np.arange(selections_hist[k])]) * np.log(
                            4 * t) / float(selections_hist[k] ** 2) * (selections_hist[k] + 1)) + np.log(4 * t) / float(
                        selections_hist[k]) / 3
                # gts and betas
                gts[t][k].append(gt)
                betas[t][k].append(beta)
                # average gts and betas
                gt = sum(gts[t][k]) / len(gts[t][k])
                beta = sum(betas[t][k]) / len(betas[t][k])
                b_k = gt + beta
                best_k_ucbs.append(b_k)
                t = trueT

            bks = list()
            for l in np.arange(L):
                max_ucb = -1
                best_k = 0
                for k, ucb in enumerate(best_k_ucbs):
                    if max_ucb < ucb and k not in bks:
                        max_ucb = ucb
                        best_k = k
                best_k_ucbs[best_k] = -1
                bks.append(best_k)

            all_acts = set()
            for l in np.arange(L):
                best_k = bks[l]
                influencer_hist_fatgt.append(INFLUENCERS[best_k])
                # observe the spread (set of activated basic nodes)
                tweet = campaign[tt * H + t][campaign[tt * H + t].influencer == INFLUENCERS[best_k]]
                acts = set()
                if not tweet.empty:
                    acts = tweet.regular_node_set_unique.values[0]
                all_acts.update(acts)

            reward = len(all_acts - prev_activated)
            prev_activated.update(all_acts)

            activations_hist_fatgt_all.append(reward)
            # GT stats
            for best_k in bks:
                activations_hist_fatgt[best_k].append(reward)
                selections_hist[best_k] += 1

    et = timer()
    elapsed_time = et - st
    return activations_hist_fatgt_all, influencer_hist_fatgt, elapsed_time


##################################################################
# LSVI-GT-UCB
##################################################################
def rl_lsvigt_reward_and_selections(seed, campaign, contexts):
    st = timer()
    c = 0.0005  # an absolute constant c > 0
    d = DCONTEXT + 1 + 1  # PHI_SIZE + 1
    paperT = T * H  # the total number of steps
    prob = 0.01  # with probability 1-p the total regret is at most
    exploration_factor = c * d * H * np.sqrt(np.log(2 * d * paperT / prob))

    influencer_hist_glm = []
    activations_hist_glm_all = []

    # the estimator of the unknown parameter theta, the Q-function, and the state and action history
    gts = dict.fromkeys(np.arange(-1, H + 1))
    betas = dict.fromkeys(np.arange(-1, H + 1))
    for h in np.arange(-1, H + 1):
        gts[h] = dict.fromkeys(np.arange(K))
        betas[h] = dict.fromkeys(np.arange(K))
        for k in np.arange(K):
            gts[h][k] = list()
            betas[h][k] = list()

    # the estimated Q-function
    Q = dict.fromkeys(np.arange(T))
    for t in np.arange(T):
        Q[t] = dict.fromkeys(np.arange(H))
        for h in np.arange(H):
            Q[t][h] = dict.fromkeys(np.arange(K), 0)

    design_matrix = dict.fromkeys(np.arange(-1, H + 1))
    for h in np.arange(-1, H + 1):
        design_matrix[h] = dict.fromkeys(np.arange(K))
        for k in np.arange(K):
            design_matrix[h][k] = np.identity(d)

    full_selections_hist = dict.fromkeys(np.arange(-1, T + 1))
    activations_hist = dict.fromkeys(np.arange(-1, T + 1))
    training_rewards = dict.fromkeys(np.arange(-1, T + 1))
    theta_estimators = dict.fromkeys(np.arange(-1, T + 1))
    rewards_hist = dict.fromkeys(np.arange(-1, T + 1))
    actions_hist = dict.fromkeys(np.arange(-1, T + 1))
    for t in np.arange(-1, T + 1):
        full_selections_hist[t] = dict.fromkeys(np.arange(-1, H + 1))
        activations_hist[t] = dict.fromkeys(np.arange(-1, H + 1))
        training_rewards[t] = dict.fromkeys(np.arange(-1, H + 1))
        theta_estimators[t] = dict.fromkeys(np.arange(-1, H + 1))
        rewards_hist[t] = dict.fromkeys(np.arange(-1, H + 1), 0)
        actions_hist[t] = dict.fromkeys(np.arange(-1, H + 1))
        for h in np.arange(-1, H + 1):
            full_selections_hist[t][h] = np.zeros(K)
            activations_hist[t][h] = np.zeros(K)
            training_rewards[t][h] = dict.fromkeys(np.arange(K))
            theta_estimators[t][h] = dict.fromkeys(np.arange(K))
            actions_hist[t][h] = np.zeros(K)
            for k in np.arange(K):
                training_rewards[t][h][k] = list()
                theta_estimators[t][h][k] = np.zeros(d)

    def phi(tau, h, k):
        context = contexts[tau * H + h]
        context = np.nan_to_num(context)
        context = preprocessing.normalize(context[:, np.newaxis], axis=0, norm='l2').ravel()
        th = h - 1
        while actions_hist[tau][th][k] == 0 and th > 0:
            th -= 1
        action = [actions_hist[tau][th][k]]
        reward = [activations_hist[tau][th][k]]
        state = np.array([*context, *action, *reward])
        state = preprocessing.normalize(state[:, np.newaxis], axis=0, norm='l2').ravel()
        return state

    def estimate_Qfunction(tau, t, h, k):
        state = phi(tau, h, k)
        inv_Sigma = np.linalg.inv(design_matrix[h][k])
        return state.dot(theta_estimators[t][h][k]), exploration_factor * np.sqrt(state.dot(inv_Sigma).dot(state))

    def take_action(tau, t, h):
        maxQ = dict.fromkeys(np.arange(K), 0)
        maxAction = np.zeros(K)

        if h >= H:
            return maxQ, maxAction

        l = 0
        actions = [i for i in product(range(2), repeat=K) if sum(i) == 1]
        actions.sort(reverse=True)
        while l != L:
            maxq = -1
            maxa = np.zeros(K)
            for action in actions:
                # if the action has already been selected
                if maxAction[action.index(1)] == 1:
                    continue
                # Qfunction and state at horizon h, state from episode tau, Qfunction estimate in episode t
                # \hat{Q}_{h,t}(s_{h,\tau},a) = \langle \phi(s_{h,\tau}, a), \hat{\theta}_{h,t} + exploration_factor * \sqrt{phi^T(s_{h,\tau},a) \Sigma^{-1}_{h,t} \phi(s_{h,\tau},a)}
                q = sum(estimate_Qfunction(tau, t, h, action.index(1)))
                if q > maxq:
                    maxq = q
                    maxa = action
            maxQ[maxa.index(1)] = maxq
            maxAction += maxa
            l += 1
        return maxQ, maxAction

    def get_training_rewards(t, h, k):
        training_rewards = list()
        for tau in np.arange(t):
            # \hat{V}_{h+1,t} = max_{a \in \mathcal{A}} \hat{Q}_{h+1,t}(s_{h+1, \tau}, a)
            maxV, maxAction = take_action(tau, t, h + 1)
            y = activations_hist[tau][h][k] / MAX_NA + maxV[k]
            training_rewards.append(y)
        return training_rewards

    for tt in np.arange(T):
        prev_activated = set()

        # rewards history for each influencer
        activations_hist_glm = dict.fromkeys(np.arange(K))
        for k in np.arange(K):
            activations_hist_glm[k] = []

        # selections per influencer
        selections_hist = dict.fromkeys(np.arange(K), 0)

        # initialization phase
        # play each influencer once#tau times
        if tt == 0:
            for k in np.arange(K):
                # observe the spread (set of activated basic nodes)
                tweet = campaign[tt * H + k][campaign[tt * H + k].influencer == INFLUENCERS[k]]
                acts = set()
                all_activations = 0
                if not tweet.empty:
                    acts = tweet.regular_node_set_unique.values[0]
                    all_activations = tweet.regular_node_count.values[0]
                reward = len(acts - prev_activated)
                prev_activated.update(acts)

                activations_hist_glm[k].append(reward)
                activations_hist[tt][k][k] = reward
                activations_hist_glm_all.append(reward)
                influencer_hist_glm.append(list())
                influencer_hist_glm[k].append(k)

                # GT stats
                # sample mean of all activations
                selections_hist[k] += 1
                full_selections_hist[tt][k + 1][k] += reward
                f = phi(tt, k, k)
                design_matrix[k][k] += np.matmul(f[np.newaxis].T, f[np.newaxis])
                actions_hist[tt][k][k] = 1

        df = pd.DataFrame()
        for h in np.arange(H - 1, -1, -1):
            for k in np.arange(K):
                training_rewards = get_training_rewards(tt, h, k)

                inv_Sigma = np.linalg.inv(design_matrix[h][k])

                x = list(
                    [phi(tau, h, k) * training_rewards[tau] if actions_hist[tau][h][k] == 1 else np.zeros(d) for tau in
                     np.arange(tt)])
                x.append(np.zeros(d))
                theta_estimators[tt][h][k] = inv_Sigma.dot(sum(x))
                theta_estimators[tt][h][k] = np.nan_to_num(theta_estimators[tt][h][k])
                theta_estimators[tt][h][k] = preprocessing.normalize(theta_estimators[tt][h][k][:, np.newaxis], axis=0,
                                                                     norm='l2').ravel()

        # UCB algorithm
        for t in np.arange(K if tt == 0 else 0, H):
            best_k_ucbs = list()
            for k in np.arange(K):
                estimated_Qfunction, qbound = estimate_Qfunction(tt, tt, t, k)
                # The GT estimator
                if selections_hist[k] == 0:
                    gt = beta = 0
                else:
                    gt = 1 / float(selections_hist[k] ** 2) * np.sum(
                        [activations_hist_glm[k][s] * s for s in np.arange(selections_hist[k])])
                    beta = (1 + np.sqrt(2)) * np.sqrt(
                        np.sum([activations_hist_glm[k][s] * s for s in np.arange(selections_hist[k])]) * np.log(
                            4 * t) / float(selections_hist[k] ** 2 * (selections_hist[k] + 1))) + np.log(4 * t) / float(
                        selections_hist[k]) / 3
                # gts and betas
                gts[t][k].append(gt)
                betas[t][k].append(beta)
                # average gts and betas
                gt = sum(gts[t][k]) / len(gts[t][k])
                beta = sum(betas[t][k]) / len(betas[t][k])

                b_k = max(gt + beta, min(H, estimated_Qfunction + qbound))
                b_k = max(gt + beta, estimated_Qfunction + qbound)

                best_k_ucbs.append(b_k)

            bks = list()
            # choose L best influencers
            for l in np.arange(L):
                max_ucb = -1
                best_k = 0
                for k, ucb in enumerate(best_k_ucbs):
                    # find the largest UCB, ignoring the already selected influencers
                    if ucb > max_ucb and k not in bks:
                        max_ucb = ucb
                        best_k = k
                # ignore best ucb in order to get the second best
                best_k_ucbs[best_k] = -1
                bks.append(best_k)

            all_activations = set()
            influencer_hist_glm.append(list())
            for l in np.arange(L):
                best_k = bks[l]
                # observe the spread (set of activated basic nodes)
                tweet = campaign[tt * H + t][campaign[tt * H + t].influencer == INFLUENCERS[best_k]]
                acts = set()
                if not tweet.empty:
                    acts = tweet.regular_node_set_unique.values[0]
                all_activations.update(acts)

            reward = len(all_activations - prev_activated)
            prev_activated.update(all_activations)

            activations_hist_glm_all.append(reward)

            # update the statistics
            for l in np.arange(L):
                best_k = bks[l]
                activations_hist[tt][t][best_k] = reward
                rewards_hist[tt][t] = reward
                activations_hist_glm[best_k].append(reward)
                actions_hist[tt][t][best_k] = 1
                selections_hist[best_k] += 1
                full_selections_hist[tt][t + 1][best_k] += reward
                f = phi(tt, t, best_k)
                design_matrix[t][best_k] += np.matmul(f[np.newaxis].T, f[np.newaxis])

    et = timer()
    elapsed_time = et - st
    return activations_hist_glm_all, influencer_hist_glm, elapsed_time



def execute_algs(seed):
    rewards_df = pd.DataFrame()

    np.random.seed(seed)

    contexts = seeds[seed]
    tweets = get_seed_tweets(contexts)

    activations_hist_lsvigt_all, influencer_hist_lsvigt, elapsed_time = rl_lsvigt_reward_and_selections(seed, tweets,
                                                                                                        contexts)
    to_add = pd.DataFrame({'reward': np.cumsum(activations_hist_lsvigt_all)})
    to_add['seed'] = seed
    to_add['algorithm'] = RLLSVIGTUCB
    to_add['round'] = to_add.index + 1
    to_add['horizon'] = H
    to_add['episode'] = T
    to_add['elapsed_time'] = elapsed_time
    rewards_df = pd.concat([rewards_df, to_add])

    activations_hist_lsvi_all, influencer_hist_lsvi, elapsed_time = lsvi_reward_and_selections(seed, tweets, contexts)
    to_add = pd.DataFrame({'reward': np.cumsum(activations_hist_lsvi_all)})
    to_add['seed'] = seed
    to_add['algorithm'] = LSVIUCB
    to_add['round'] = to_add.index + 1
    to_add['horizon'] = H
    to_add['episode'] = T
    to_add['elapsed_time'] = elapsed_time
    rewards_df = pd.concat([rewards_df, to_add])

    activations_hist_lsvi_all, influencer_hist_lsvi, elapsed_time = lsvi_reward_and_selections_septhetas(seed, tweets,
                                                                                                         contexts)
    to_add = pd.DataFrame({'reward': np.cumsum(activations_hist_lsvi_all)})
    to_add['seed'] = seed
    to_add['algorithm'] = LSVIUCBST
    to_add['round'] = to_add.index + 1
    to_add['horizon'] = H
    to_add['episode'] = T
    to_add['elapsed_time'] = elapsed_time
    rewards_df = pd.concat([rewards_df, to_add])

    activations_hist_fatgt_all, influencer_hist_fatgt, elapsed_time = fatgt_reward_and_selections(seed, tweets,
                                                                                                  contexts)
    to_add = pd.DataFrame({'reward': np.cumsum(activations_hist_fatgt_all)})
    to_add['seed'] = seed
    to_add['algorithm'] = FATGTUCB
    to_add['round'] = to_add.index + 1
    to_add['horizon'] = H
    to_add['episode'] = T
    to_add['elapsed_time'] = elapsed_time
    rewards_df = pd.concat([rewards_df, to_add])

    activations_hist_rlfatgt_all, influencer_hist_rlfatgt, elapsed_time = rlfatgt_reward_and_selections(seed, tweets,
                                                                                                        contexts)
    to_add = pd.DataFrame({'reward': np.cumsum(activations_hist_rlfatgt_all)})
    to_add['seed'] = seed
    to_add['algorithm'] = RLFATGTUCB
    to_add['round'] = to_add.index + 1
    to_add['horizon'] = H
    to_add['episode'] = T
    to_add['elapsed_time'] = elapsed_time
    rewards_df = pd.concat([rewards_df, to_add])

    return rewards_df

rewards_df = pd.DataFrame()

for l in [1, 2, 5]:

    T = 50
    H = 30
    L = l
    BUDGET = T * H

    result = Parallel(n_jobs=NUM_CORES)(
        delayed(execute_algs)(seed) for seed in seeds
    )
    rewards_df = rewards_df.append(result)
    rewards_df = rewards_df.set_index(np.arange(len(rewards_df)))

    rewards_df.to_pickle("lsvi" + str(L) + ".pkl")

    for h in rewards_df.horizon.unique():
        plt.clf()
        df2 = rewards_df[rewards_df.horizon == h]
        sns.lineplot(x='round', y='reward',
                     hue='algorithm',
                     style="algorithm", err_style="band",
                     dashes=True,
                     data=df2)
        plt.xlabel("rounds")
        plt.ylabel("cumulative spread")
        plt.title("Twitter data - L=" + str(L))
        plt.savefig("RL_twitter_" + str(L) + "influencers.pdf", bbox_inches='tight')

        plt.clf()
        df2 = rewards_df[rewards_df.horizon == h]
        sns.lineplot(x='round', y='reward',
                     hue='algorithm',
                     style="algorithm", err_style="band",
                     dashes=True,
                     data=df2)
        plt.xlabel("rounds")
        plt.ylabel("cumulative spread")
        plt.xlim(left=1400, right=1500)
        plt.ylim(bottom=rewards_df.loc[rewards_df['round'] == 1400].min().reward)
        plt.title("Twitter data - L=" + str(L))
        plt.savefig(str(L) + "twitter_zoomed.pdf", bbox_inches='tight')

'''
