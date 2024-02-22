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
maps = {"cong", "lower", "upper"}

for districting_map in maps:
    EPSILON = 0.05 if districting_map == "cong" else 0.11
        
    for state in states:
        if state == "AK":
            data = pd.read_csv(file_prefix + '{}/ensemble/{}ensemble{}{:.1%}{}.csv'.format(state+"lower", state+"lower", state+"lower", 0.11, length))
        else:
            data = pd.read_csv(file_prefix + '{}/ensemble/{}ensemble{}{:.1%}{}.csv'.format(state+districting_map, state+districting_map, state+districting_map, EPSILON, length))
    
        for column in scores_column_names:
           # Your data
            data_values = data[column]
            
            # Choose the desired bin width
            #desired_bin_width = 0.08  # Adjust as needed
            
            # Calculate the number of bins based on the desired bin width
            #num_bins = max(int((max(data_values) - min(data_values)) / desired_bin_width), 1)  # Ensure num_bins is at least 1
            
            # num_bins = min(30, len(set(data_values)))
            # desired_bin_width = 0.08
            # # Calculate bin edges manually centered on data values
            # bin_centers = np.array(sorted(np.linspace(min(data_values) + desired_bin_width/2, max(data_values) - desired_bin_width/2, num_bins)))
            # bin_edges = np.concatenate((np.array([min(data_values) - desired_bin_width/2]), bin_centers + desired_bin_width/2))
            
            if len(set(data_values)) < 10:
                num_bins = len(set(data_values))
                bin_values = set(data_values)
                print(bin_values)
                #desired_bin_width = 0.08
                # Calculate bin edges manually centered on data values
                bin_centers = np.array(sorted(bin_values))
                differences = [bin_centers[i]-bin_centers[i-1] for i in range(1, len(bin_centers))]
                print(bin_centers)
                print(differences)
                min_difference = min(differences)
                if len(bin_centers) < 2:
                    print(column)
                bin_edges = [bin_centers[0]-0.05, bin_centers[0]+0.05] if len(bin_centers) < 2 else np.concatenate(([bin_centers[0]-min_difference/2], bin_centers + min_difference/2))
                
            else:
                bin_edges= 30
            
            
            # Plot histogram with centered bins and wider bin width
            my_plot = plt.hist(data_values, bins =bin_edges, align = "mid",  edgecolor='black')
            
            # Plot a dot above the bin for the specific value
            value_to_plot = data_values[0]
            #closest_index = np.abs(my_plot[1] - value_to_plot).argmin()
            plt.scatter([value_to_plot], [0.02*max(my_plot[0])],
                        color='red', marker='o', label=f'Value for {state} : {value_to_plot}')
            plt.xlabel('{}'.format(column))
            plt.ylabel('Frequency')
            # Show the plot
            plt.legend()
            
            plt.savefig('./data/results/{}_ensembles/ensemble{}{:.1%}{}{}.png'.format(districting_map, state+districting_map, EPSILON, length, column))
            plt.show()




