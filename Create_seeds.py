"""
Thursday Feb 25, 2021


Using the ReCom proposal following https://gerrychain.readthedocs.io/en/latest/user/recom.html

Authors: Tommy Ratliff, Ellen Veomett

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
from helpers import geo
from gerrychain.proposals import recom
from functools import partial
import pandas
import geopandas as gpd
import csv
import os
import datetime  #For keeping track of runtime
import pandas as pd
import json



"""
-------------- Only 4 parameters that should need changing -----------------
Can also update initial districting plan by changing 'assignment=' in the initial_partition

Make sure to update election results to be written into districts files

"""

beginrun = datetime.datetime.now()
print ("\nBegin date and time : ", beginrun.strftime("%Y-%m-%d %H:%M:%S"))


outdir="./PA_seed/"
file_prefix = "PA"

election_name = "SEN16"

total_steps_in_run=500
save_district_graph_mod=1
save_district_plot_mod=100

os.makedirs(outdir, exist_ok=True)
#graph = Graph.from_file("./PA.shp")
graph = Graph.from_json("./PA.json")

elections = [
    Election("SEN10", {"Democratic": "SEN10D", "Republican": "SEN10R"}),
    Election("SEN12", {"Democratic": "USS12D", "Republican": "USS12R"}),
    Election("SEN16", {"Democratic": "T16SEND", "Republican": "T16SENR"}),
    Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
    Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
]
# Population updater, for computing how close to equality the district
# populations are. "TOT_POP" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = GeographicPartition(graph, 
                                        assignment= "CD_2011", #"2011_PLA_1",     # "GOV", "REMEDIAL_P", 
                                        updaters=my_updaters)

# The ReCom proposal needs to know the ideal population for the districts so that
# we can improve speed by bailing early on unbalanced partitions.

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

# We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal.
proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                  )

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)




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




df=gpd.read_file("./PA.shp")


    
#Run through chain, building 
for t, part in enumerate(chain):
    geo_score = abs(geo(part, election_name)[0]-geo(part, election_name)[1])  # difference in geo scores
    eg_score = abs(part[election_name].efficiency_gap())  #absolute value of efficiency gap
    mm_score = abs(part[election_name].mean_median())  # absolute value of mean-median
    if geo_score <=2 and eg_score <= 0.08 and mm_score <=0.08:
        print("found it!")
        print("GEO is ", geo_score, " EG is ", eg_score, " MM is ", mm_score)
        
        # export graph of this partition to json file
        (part.graph).to_json(outdir + file_prefix + "seed.json")
        
        # Create the assignment for this partition
        seed_dict = dict()
        seed_nodes = list(part.graph.nodes)
        for node in seed_nodes:
            seed_dict[node] = part.assignment[node]
        with open("./PA_seed/PAassignment.json", "w") as outfile:
            json.dump(seed_dict, outfile)
        
        #Create plot of this partition and export
        df.plot(pandas.Series(part.assignment), cmap="tab20", figsize=(16,8)) 
        plot_output_file = outdir + file_prefix + "seed_plot.png" # export plot
        plt.savefig(plot_output_file)
        plt.close()
        break
    else:
        print("GEO is ", geo_score, " EG is ", eg_score, " MM is ", mm_score)

            

endrun = datetime.datetime.now()
print ("\nEnd date and time : ", endrun.strftime("%Y-%m-%d %H:%M:%S"))

    
diff=endrun-beginrun
print("\nTotal time: ", str(diff))

    