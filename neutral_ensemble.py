"""
Thursday Feb 25, 2021


Using the ReCom proposal following https://gerrychain.readthedocs.io/en/latest/user/recom.html

Authors: Tommy Ratliff, Ellen Veomett.  Edited by Ellen Veomett for the purpose of creating seed plans.

For reproducability of chain:
    
See https://gerrychain.readthedocs.io/en/latest/topics/reproducibility.html
for setting the random.seed for gerrychain 
and https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables
for setting the environment variable PYTHONHASHSEED=0  

***Seems to work to always produce same chain. Change random.seed to get different one***

"""
from gerrychain.random import random
random.seed(12345678)

import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from gerrychain.metrics.compactness import polsby_popper
from functools import partial
import pandas
import geopandas as gpd
import csv
import os
import datetime  #For keeping track of runtime
import pandas as pd
import json
from pathlib import Path # to create directory if needed
from helpers import geo, declination_1, declination_false, std_dev_vote_shares, std_dev_neighborhood_ave, state_v

"""
-------------- parameters that should need changing -----------------
outdir, file_prefix, election_name, graph, elections, assignment (in initial_partition)
Can also update initial districting plan by changing 'assignment=' in the initial_partition

"""

state = "AK"
districted_map = "lower"
election_name = "PRES16"
EPSILON = 0.11
total_steps_in_run=20000

num_h_districts = {"PAcong": 18, "PAupper": 50, "PAlower": 203, "MAcong": 9, "MAupper": 40, "MAlower": 160, "TXcong": 36, "TXlower": 150, "TXupper": 31, "OKcong": 5,
                   "OKlower": 101, "OKupper": 48, "MIcong": 13, "MIupper": 38, "MIlower": 110, "ORcong": 5, "ORupper": 30, "ORlower": 60, "AKlower": 40}

total_seats = num_h_districts[state+districted_map]

beginrun = datetime.datetime.now()
print ("\nBegin date and time : ", beginrun.strftime("%Y-%m-%d %H:%M:%S"))

seed_location_prefix = "./data/seeds/{}_permissive/".format(state)
outdir="./data/results/{}{}/ensemble/".format(state, districted_map)
file_prefix = "{}{}".format(state, districted_map)

Path(outdir).mkdir(parents=True, exist_ok=True)


save_district_graph_mod=1
save_district_plot_mod=100


#graph = Graph.from_file("./PA.shp")
#graph = Graph.from_file(seed_location_prefix + "mi16_results.shp")
#graph = Graph.from_json(seed_location_prefix + "wisconsin2011_dualgraph.json")
#graph = Graph.from_json(seed_location_prefix + "PA.json")
#graph = Graph.from_file(seed_location_prefix + "MA_precincts_12_16.shp")
#graph = Graph.from_file(seed_location_prefix + "WI.shp")
#graph = Graph.from_json(seed_location_prefix + "TX.json")
#graph = Graph.from_file(seed_location_prefix + "OK_precincts.shp")

graphname = seed_location_prefix + "{}_seed/{}seed.json".format(state+districted_map, state + districted_map,)
graph = Graph.from_json(graphname)

#print("graph nodes are", graph.nodes)
#elections = [Election("SEN14", {"Democratic": "SEN14D", "Republican": "SEN14R"})]
#elections = [Election("GOV18", {"Democratic": "GOV18D", "Republican": "GOV18R"})]
#elections = [Election("SEN18", {"Democratic": "SEN18D", "Republican": "SEN18R"})]
#elections = [Election("G18GOV", {"Democratic": "G18GOVD", "Republican": "G18GOVR"})]
#elections = [Election("T16SEN", {"Democratic": "T16SEND", "Republican": "T16SENR"})]
#elections = [Election("SSEN16", {"Democratic": "SSEN16D", "Republican": "SSEN16R"})]
elections = [Election("PRES16", {"Democratic": "PRES16D", "Republican": "PRES16R"})]

# elections = [
#     Election("SEN10", {"Democratic": "SEN10D", "Republican": "SEN10R"}),
#     Election("SEN12", {"Democratic": "USS12D", "Republican": "USS12R"}),
#     Election("SEN16", {"Democratic": "T16SEND", "Republican": "T16SENR"}),
#     Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
#     Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
# ]


# Population updater, for computing how close to equality the district
# populations are. "TOT_POP" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = GeographicPartition(graph, 
                                        assignment= "HDIST", #"2011_PLA_1",     # "GOV", "REMEDIAL_P", 
                                        updaters=my_updaters)



num_districts = len(initial_partition)
print("the number of districts we got was: ", num_districts)

# The ReCom proposal needs to know the ideal population for the districts so that
# we can improve speed by bailing early on unbalanced partitions.

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
#print(initial_partition["population"].values())

# We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal.
proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=EPSILON,
                   node_repeats=2
                  )

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, EPSILON)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=total_steps_in_run
    )

count = 0

#data frame to hold metric scores
scores_column_names = ["Target Column Districts Won", "GEO score ratio", "GEO Dem", "GEO Rep", "Efficiency Gap with wasted votes",
                       "Efficiency Gap with S, V", "Mean-Median", "Declination", "Polsby Popper Average", "Polsby Popper Min"]
all_scores_df = pd.DataFrame(columns = scores_column_names)


#Run through chain, building 
for t, part in enumerate(chain):
    if t%200 == 0:
        print("At step ", t, " the time so far is ", datetime.datetime.now() - beginrun)
    dem_seats_won = part[election_name].seats("Democratic")
    all_scores_df.at[t, "Target Column Districts Won"] = dem_seats_won
    geo_score = geo(part, election_name)
    all_scores_df.at[t, "GEO Dem"] = geo_score[0]
    all_scores_df.at[t, "GEO Rep"] = geo_score[1]
    all_scores_df.at[t, "GEO score ratio"] = (geo_score[1] - geo_score[0])/total_seats  #changed to correct sign
    all_scores_df.at[t, "Efficiency Gap with wasted votes"] = part[election_name].efficiency_gap()
    V = sum(part[election_name].votes("Democratic"))/part[election_name].total_votes()
    S = dem_seats_won/total_seats
    all_scores_df.at[t, "Efficiency Gap with S, V"] = S-2*V+1/2
    all_scores_df.at[t, "Mean-Median"] = part[election_name].mean_median()
    pp_dict = polsby_popper(part)
    all_scores_df.at[t, "Polsby Popper Average"] = sum(pp_dict.values())/len(pp_dict)
    all_scores_df.at[t, "Polsby Popper Min"] = min(pp_dict.values())
    all_scores_df.at[t, "Declination"] = declination_false(part, election_name)

endrun = datetime.datetime.now()
print ("\nEnd date and time : ", endrun.strftime("%Y-%m-%d %H:%M:%S"))

df_out = outdir+file_prefix+"ensemble{}{}{:.1%}{}.csv".format(state, districted_map,  EPSILON, total_steps_in_run)
all_scores_df.to_csv(df_out, header=True) 
    
diff=endrun-beginrun
print("\nTotal time: ", str(diff))

    