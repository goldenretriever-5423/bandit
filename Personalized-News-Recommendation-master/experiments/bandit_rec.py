#-*-coding:utf-8-*-

# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('..')
from bandits import *
from evaluator import evaluate
from matplotlib import pyplot as plt
import pandas as pd


# %%time
import dataset
files = ("/media/yuting/TOSHIBA EXT/yahoo/R6/ydata-fp-td-clicks-v1_0.20090502")
dataset.get_yahoo_events(files)

tests = [LinUCB(0.3, context="both")]

for test in tests:
    _,_ = evaluate(test)