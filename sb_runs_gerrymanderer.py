#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:29:27 2023

@author: Ellen Veomett, and all authors of https://github.com/vrdi/shortbursts-gingles/blob/main/state_experiments/sb_runs.py
"""

import argparse
import geopandas as gpd
import numpy as np
import pickle
from functools import partial
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges, perimeter, exterior_boundaries, boundary_nodes, interior_boundaries, cut_edges_by_part
from gerrychain import MarkovChain
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain import constraints
from gerrychain.tree import recursive_tree_part
from gerrymanderer import Gingleator
from gerrychain.updaters import election
#from little_helpers import *
import json
from pathlib import Path # to create directory if needed

## Read in 
"""
Ellen's note: This tells us what to type when running the short burst.  
wFor example: python sb_runs_gerrymanderer.py PA cong 500 10 T16SEND 0
means we'll run the PA map, the congressional map, 500 steps, length of burst is 10 (so 50 bursts), use the 
T16Senate Democratic column, and use score function labeled 0 (see score_functs below)

python sb_runs_gerrymanderer.py MA cong 500 10 SEN18D 0  

python sb_runs_gerrymanderer.py TX cong 500 10 SEN14D 0  

"""
parser = argparse.ArgumentParser(description="SB Chain run", 
                                 prog="sb_runs_gerrymanderer.py")
parser.add_argument("state", metavar="state_id", type=str,
                    choices=["PA", "MA", "TX"],
                    help="which state to run chains on")
parser.add_argument("map", metavar = "map_type", type = str,
                    choices = ["cong", "lower", "upper"],
                    help = "which map is it, cong, lower, or upper")
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

#String below tells whether we want to restrict GEO, EG, or mean-median
#Probably these should eventually be arguments, as above
#METRIC = "GEO"
#METRIC = "EG"
#METRIC = "MM"
METRIC = None
BIAS = False



num_h_districts = {"PAcong": 18, "MAcong": 9, "MAupper": 40, "MAlower": 160, "TXcong": 36, "TXlower": 150, "TXupper": 31}


score_functs = {0: None, 1: Gingleator.reward_partial_dist, 
                2: Gingleator.reward_next_highest_close,
                3: Gingleator.penalize_maximum_over,
                4: Gingleator.penalize_avg_over}

BURST_LEN = args.l
NUM_DISTRICTS = num_h_districts[args.state + args.map]
ITERS = args.iters
POP_COL = "TOTPOP"
N_SAMPS = 10
SCORE_FUNCT = None #score_functs[args.score]
EPS = 0.05
TARGET_POP_COL = args.col
ELECTION = args.col[:-1]  #remove the party name


## Setup graph, updaters, elections, and initial partition

print("Reading in Data/Graph", flush=True)

graphname = "./data/seeds/{}/{}_seed/{}seed.json".format(args.state, args.state + args.map, args.state + args.map)
graph = Graph.from_json(graphname)

#NEW STUFF BELOW
# elections = [
#     Election("SEN10", {"Democratic": "SEN10D", "Republican": "SEN10R"}),
#     Election("SEN12", {"Democratic": "USS12D", "Republican": "USS12R"}),
#     Election("SEN16", {"Democratic": "T16SEND", "Republican": "T16SENR"}),
#     Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
#     Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
# ]
#NEW STUFF ABOVE
#elections = [Election("SEN18", {"Democratic": "SEN18D", "Republican": "SEN18R"})]
#elections = [Election("SEN14", {"Democratic": "SEN14D", "Republican": "SEN14R"})]

#Note: The stuff above and below is redundant right now and should be fixed/condensed at some point
# my_updaters = {"population" : Tally(POP_COL, alias="population"),
#                 "VAP": Tally("VAP"),
#                 "BVAP": Tally("BVAP"),
#                 "HVAP": Tally("HVAP"),
#                 "WVAP": Tally("WVAP"),
#                 "T16SENR": Tally("T16SENR"), #added
#                 "T16SEND": Tally("T16SEND"), #added
#                 "T16SEN": Election(
#                             "2016 Senate",
#                             {"Democratic": "T16SEND", "Republican": "T16SENR"},
#                             alias="T16SEN" #added to do eg . . .
#                             ),
#                 "PRES16R": Tally("PRES16R"),
#                 #"ELECTION": election,
#                 "nWVAP": lambda p: {k: v - p["WVAP"][k] for k,v in p["VAP"].items()},
#                 "cut_edges": cut_edges}
# my_updaters = {"population" : Tally(POP_COL, alias="population"),
#                "SEN18R": Tally("SEN18R"), #added
#                "SEN18D": Tally("SEN18D"), #added
#                "SEN18": Election(
#                            "2018 Senate",
#                            {"Democratic": "SEN18D", "Republican": "SEN18R"},
#                            alias="SEN18" #added to do eg . . .
#                            ),
#                "cut_edges": cut_edges}
# my_updaters = {"population" : Tally(POP_COL, alias="population"),
#                "SEN14R": Tally("SEN14R"), #added
#                "SEN14D": Tally("SEN14D"), #added
#                "SEN14": Election(
#                            "2014 Senate",
#                            {"Democratic": "SEN14D", "Republican": "SEN14R"},
#                            alias="SEN14" #added to do eg . . .
#                            ),
#                "cut_edges": cut_edges}
elections = [Election(ELECTION, {"Democratic": ELECTION+"D", "Republican": ELECTION + "R"})]
my_updaters = {"population" : Tally(POP_COL, alias="population"),
                ELECTION+"R": Tally(ELECTION+"R"), #added
                ELECTION+"D": Tally(ELECTION+"D"), #added
                ELECTION : Election(
                            ELECTION,
                            {"Democratic": ELECTION + "D", "Republican": ELECTION + "R"},
                            alias=ELECTION #added to do eg . . .
                            ),
                "cut_edges": cut_edges,
                "perimeter": perimeter,
                "area": Tally("area", alias="area"),
                "exterior_boundaries": exterior_boundaries, 
                "boundary_nodes": boundary_nodes,
                "interior_boundaries": interior_boundaries,
                "cut_edges_by_part": cut_edges_by_part} # area, perimiter, exterior_boundaries, boundary_nodes needed for polsby popper score




print("Creating seed plan", flush=True)


print("using " + ELECTION + " election")

total_pop = sum([graph.nodes()[n][POP_COL] for n in graph.nodes()])

seed_bal = {"AR": "05", "CO": "02", "LA": "04", "NM": "04", "TX": "02", "VA": "02"}


##Below is from sb_runs
with open("./data/seeds/{}/{}_seed/{}seed_assignment.json".format(args.state, args.state + args.map, args.state + args.map), "r") as f:
    cddict = json.load(f)

cddict = {int(k):v for k,v in cddict.items()}

init_partition = Partition(graph, assignment=cddict, updaters=my_updaters)
## Above is from sb_runs

"""  Only use this when starting from gerrymandered map
init_partition = GeographicPartition(graph, 
                                        assignment= "CD_2011", #"2011_PLA_1",     # "GOV", "REMEDIAL_P", 
                                        updaters=my_updaters)
"""

gingles = Gingleator(init_partition, num_districts = NUM_DISTRICTS, pop_col=POP_COL,
                     threshold=0.5, score_funct=SCORE_FUNCT, epsilon=EPS,
                     target_perc_col="{}_perc".format(TARGET_POP_COL), election_name = ELECTION)

"""
The if/elseif commands below make sure that if we want to maximize D votes, we devide by D+R votes (same for R)
If we want to maximize minority votes, we divide by VAP
"""
last_char = TARGET_POP_COL[-1]
if last_char == "D":
    denominator_col = TARGET_POP_COL[:-1] + "R"
elif last_char == "R":
    denominator_col = TARGET_POP_COL[:-1] + "D"
else:
    denominator_col = "VAP"
    
gingles.init_target_perc_col(TARGET_POP_COL, denominator_col, 
                               "{}_perc".format(TARGET_POP_COL))

num_bursts = int(ITERS/BURST_LEN)

print("Starting Short Bursts Runs", flush=True)

#NOTE: BELOW IS WHAT CHANGES THE METRIC IN OUR RUN!  Also if BIAS = true, it's a biased run.  Otherwise the metric is *required* to be within particular bounds
for n in range(N_SAMPS):
    print("Metric chosen is ", METRIC)
    #If, elif below accounts for changing metric/bias
    if METRIC == "GEO":
        if BIAS:
            print("Performing a biased short burst run using the GEO metric")
            sb_obs = gingles.geo_biased_short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
        else: 
            print("Performing a short burst run with GEO metric restricted")
            sb_obs = gingles.geo_short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
    elif METRIC == "EG":
        if BIAS:
            print("Performing a biased short burst run using the Efficiency Gap")
            sb_obs = gingles.eg_biased_short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
        else:
            print("Performing a short burst run with Efficiency Gap restricted")
            sb_obs = gingles.eg_short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
    elif METRIC == "MM":
        if BIAS:
            print("Performing a biased short burst run using the Mean Median difference")
            sb_obs = gingles.mm_biased_short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                             maximize=True, verbose=False)
        else:
            print("Performing a short burst run with Mean Median difference restricted")
            sb_obs = gingles.mm_short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                             maximize=True, verbose=False)
    else:
        print("Doing a short burst while evaluating all metrics.")
        sb_obs = gingles.short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
    print("\tFinished chain {}".format(n), flush=True)

    print("\tSaving results", flush=True)
    
    Path("data/results/{}/{}/".format(args.state + args.map, METRIC)).mkdir(parents=True, exist_ok=True) # In case a directory doesn't exist

    f_out = "data/results/{}/{}/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_score{}_{}_bias{}_{}.npy".format(args.state + args.map, METRIC, args.state + args.map,
                                                        NUM_DISTRICTS, TARGET_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, args.score, METRIC, BIAS, n)
    np.save(f_out, sb_obs[1])

    f_out_part = "data/results/{}/{}/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_score{}_{}_bias{}_{}_max_part.p".format(args.state + args.map, METRIC, args.state + args.map,
                                                        NUM_DISTRICTS, TARGET_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, args.score, METRIC, BIAS, n)

    max_stats = {#"VAP": sb_obs[0][0]["VAP"],
                 #"BVAP": sb_obs[0][0]["BVAP"],
                 #"WVAP": sb_obs[0][0]["WVAP"],
                 #"HVAP": sb_obs[0][0]["HVAP"],
                 ELECTION + "R": sb_obs[0][0][ELECTION + "R"],
                 ELECTION + "D": sb_obs[0][0][ELECTION + "D"]}

    with open(f_out_part, "wb") as f_out:
        pickle.dump(max_stats, f_out)
        
    if METRIC == None:
        df_out = "data/results/{}/{}/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_score{}_{}_bias{}_{}.csv".format(args.state + args.map, METRIC, args.state + args.map,
                                                            NUM_DISTRICTS, TARGET_POP_COL, EPS, 
                                                            ITERS, BURST_LEN, args.score, METRIC, BIAS, n)
        sb_obs[2].to_csv(df_out, header=True) 
        