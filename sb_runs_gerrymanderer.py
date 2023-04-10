#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:29:27 2023

@author: erv2, and all authors of https://github.com/vrdi/shortbursts-gingles/blob/main/state_experiments/sb_runs.py
"""

import argparse
import geopandas as gpd
import numpy as np
import pickle
from functools import partial
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain import constraints
from gerrychain.tree import recursive_tree_part
from gerrymanderer import Gingleator
from gerrychain.updaters import election
#from little_helpers import *
import json

## Read in 
"""
Ellen's note: This tells us what to type when running the short burst.  
For example: python sb_runs_gerrymanderer.py PA 500 10 T16SEND 0
means we'll run the PA map, 500 steps, length of burst is 10 (so 50 bursts), use the 
T16Senate Democratic column, and use score function labeled 0 (see score_functs below)
"""
parser = argparse.ArgumentParser(description="SB Chain run", 
                                 prog="sb_runs_gerrymanderer.py")
parser.add_argument("state", metavar="state_id", type=str,
                    choices=["VA", "TX", "AR", "CO", "LA", "NM", "PA"],
                    help="which state to run chains on")
parser.add_argument("iters", metavar="chain_length", type=int,
                    help="how long to run each chain")
parser.add_argument("l", metavar="burst_length", type=int,
                    help="The length of each short burst")
parser.add_argument("col", metavar="column", type=str,
                    help="Which column to optimize")
parser.add_argument("score", metavar="score_function", type=int,
                    help="How to count gingles districts",
                    choices=[0,1,2,3,4])
args = parser.parse_args()

num_h_districts = {"VA": 100, "TX": 150, "AR": 100, "CO": 65, "LA": 105, "NM": 70, "PA": 18}


score_functs = {0: None, 1: Gingleator.reward_partial_dist, 
                2: Gingleator.reward_next_highest_close,
                3: Gingleator.penalize_maximum_over,
                4: Gingleator.penalize_avg_over}

BURST_LEN = args.l
NUM_DISTRICTS = num_h_districts[args.state]
ITERS = args.iters
POP_COL = "TOTPOP"
N_SAMPS = 1
SCORE_FUNCT = None #score_functs[args.score]
EPS = 0.045
MIN_POP_COL = args.col


## Setup graph, updaters, elections, and initial partition

print("Reading in Data/Graph", flush=True)

#graph = Graph.from_json("PA.json")
graph = Graph.from_json("PA.json")

#NEW STUFF BELOW
elections = [
    Election("SEN10", {"Democratic": "SEN10D", "Republican": "SEN10R"}),
    Election("SEN12", {"Democratic": "USS12D", "Republican": "USS12R"}),
    Election("SEN16", {"Democratic": "T16SEND", "Republican": "T16SENR"}),
    Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
    Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
]
#NEW STUFF ABOVE

my_updaters = {"population" : Tally(POP_COL, alias="population"),
               "VAP": Tally("VAP"),
               "BVAP": Tally("BVAP"),
               "HVAP": Tally("HVAP"),
               "WVAP": Tally("WVAP"),
               "T16SENR": Tally("T16SENR"), #added
               "T16SEND": Tally("T16SEND"), #added
               "T16SEN": Tally("T16SEND", "T16SENR"), #added
               "PRES16R": Tally("PRES16R"),
               #"ELECTION": election,
               "nWVAP": lambda p: {k: v - p["WVAP"][k] for k,v in p["VAP"].items()},
               "cut_edges": cut_edges}


print("Creating seed plan", flush=True)

total_pop = sum([graph.nodes()[n][POP_COL] for n in graph.nodes()])

seed_bal = {"AR": "05", "CO": "02", "LA": "04", "NM": "04", "TX": "02", "VA": "02"}

#with open("PA.json", "r") as f:
#    cddict = json.load(f)

#cddict = {int(k):v for k,v in cddict.items()}

#init_partition = Partition(graph, assignment=cddict, updaters=my_updaters)
init_partition = GeographicPartition(graph, 
                                        assignment= "CD_2011", #"2011_PLA_1",     # "GOV", "REMEDIAL_P", 
                                        updaters=my_updaters)


gingles = Gingleator(init_partition, pop_col=POP_COL,
                     threshold=0.5, score_funct=SCORE_FUNCT, epsilon=EPS,
                     minority_perc_col="{}_perc".format(MIN_POP_COL))

"""
Ellen's comment: Note that I edited below.  This is because VAP isn't what we want to divide by!
Probably we should fix this up a bit later.
"""
gingles.init_minority_perc_col(MIN_POP_COL, "T16SENR", 
                               "{}_perc".format(MIN_POP_COL))

#BELOW IS NEW SO WE CAN GET TOTAL SEATS
gingles.init_total_seats(num_h_districts[args.state])

num_bursts = int(ITERS/BURST_LEN)

print("Starting Short Bursts Runs", flush=True)

#NOTE WE"RE TRYING OUT AN EG BASED RUN BELOW!!!!!
for n in range(N_SAMPS):
    sb_obs = gingles.eg_biased_short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
    print("\tFinished chain {}".format(n), flush=True)

    print("\tSaving results", flush=True)

    f_out = "data/states/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_score{}_{}.npy".format(args.state,
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, args.score, n)
    np.save(f_out, sb_obs[1])

    f_out_part = "data/states/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_score{}_{}_max_part.p".format(args.state,
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, args.score, n)

    max_stats = {"VAP": sb_obs[0][0]["VAP"],
                 "BVAP": sb_obs[0][0]["BVAP"],
                 "WVAP": sb_obs[0][0]["WVAP"],
                 "HVAP": sb_obs[0][0]["HVAP"],
                 "T16SENR": sb_obs[0][0]["T16SENR"]}

    with open(f_out_part, "wb") as f_out:
        pickle.dump(max_stats, f_out)