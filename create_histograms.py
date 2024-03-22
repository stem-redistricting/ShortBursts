#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:39:33 2024

@author: eveomett
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_prefix = "./data/results/"
length = 20000


states = {"MA", "PA", "TX", "OK", "MI", "OR", "AK"}
scores_column_names = ["Target Column Districts Won", "GEO score ratio", "GEO Dem", "GEO Rep", "Efficiency Gap with wasted votes",
                       "Efficiency Gap with S, V", "Mean-Median", "Declination", "Polsby Popper Average", "Polsby Popper Min"]
scores_hist_titles = {"Target Column Districts Won": "Democratic Districts won", "GEO score ratio": "GEO score ratio", "GEO Dem": "Democratic GEO score",
                      "GEO Rep": "Republican GEO score", "Efficiency Gap with wasted votes": "Efficiency Gap",
                       "Efficiency Gap with S, V": "Efficiency Gap with S, V", "Mean-Median": "Mean-Median Difference", "Declination": "Declination",
                       "Polsby Popper Average": "Polsby Popper Average", "Polsby Popper Min": "Polsby Popper Min"}
maps = {"cong", "lower", "upper"}

for districting_map in maps:
    EPSILON = 0.05 if districting_map == "cong" else 0.11
        
    for state in states:
        if state == "AK":
            data = pd.read_csv(file_prefix + '{}/ensemble/{}ensemble{}{:.1%}{}.csv'.format(state+"lower", state+"lower", state+"lower", 0.11, length))
        else:
            data = pd.read_csv(file_prefix + '{}/ensemble/{}ensemble{}{:.1%}{}.csv'.format(state+districting_map, state+districting_map, state+districting_map, EPSILON, length))
    
        for column in scores_column_names:
            data_values = data[column]
            

            if len(set(data_values)) < 20:
                num_bins = len(set(data_values))
                bin_values = set(data_values)
                print(bin_values)

                bin_centers = np.array(sorted(bin_values))
                differences = [bin_centers[i]-bin_centers[i-1] for i in range(1, len(bin_centers))]
                print(bin_centers)
                print(differences)
                min_difference = min(differences)
                if len(bin_centers) < 2:
                    print(column)
                bin_edges = [bin_centers[0]-0.05, bin_centers[0]+0.05] if len(bin_centers) < 2 else np.concatenate(([bin_centers[0]-min_difference/2], bin_centers + min_difference/2))
                
                
            else:
                bin_edges= 20
            
            

            my_plot = plt.hist(data_values, bins =bin_edges, align = "mid",  edgecolor='black')
            

            value_to_plot = data_values[0]

            plt.scatter([value_to_plot], [0.02*max(my_plot[0])],
                        color='red', marker='o', label=f'Value for {state} : {value_to_plot}')
            plt.xlabel('{}'.format(scores_hist_titles[column]))
            plt.ylabel('Frequency')
  
            plt.legend()
            
            plt.savefig('./data/results/{}_ensembles/ensemble{}{}eps_{}{}.png'.format(districting_map, state+districting_map, int(EPSILON*100), length, column))
            plt.show()




