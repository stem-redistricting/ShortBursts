#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:25:02 2023

@author: erv2
"""

import pandas as pd

"""
GEO metric below.  Right now the election is hard coded.  Should change later.
"""

def geo(part, election):
    """

    Parameters
    ----------
    part : the partition

    Returns
    -------
    Geo score for that map as a list
    This is defined as a function on a partition so that we can use it in the constraints list
    """
    
    # First find the set of edges from the partition, put into a dataframe
    # This first bit comes from our recom code
    edges_set=set()
    for e in part["cut_edges"]:
        edges_set.add( (part.assignment[e[0]],part.assignment[e[1]] ))
    edges_list = list(edges_set)
    edges_df = pd.DataFrame(edges_list)
    
    # Clean up the dataframe (comes from GEO code)
    tmp_df = edges_df.rename(columns={0:1,1:0},copy=False)
    all_edges_df = pd.concat([edges_df,tmp_df])
    all_edges_df.drop_duplicates( keep='first', inplace=True)
    all_edges_df.reset_index(drop=True, inplace=True)  
    
    # Create Election Dataframe
    # NOTE: THE STUFF BELOW IS HARD CODED AND SHOULD BE CHANGED TO ACCOUNT FOR SPECIFIC ELECTION
    
    D_votes = part[election].votes("Democratic")
    R_votes = part[election].votes("Republican")
    dist_num = [i+1 for i in range(len(D_votes))]
    election_df = pd.DataFrame(list(zip(D_votes, R_votes)), index = dist_num, columns =[1,2])
    
    num_parties = len(election_df.columns)            
    total_votes =  election_df.iloc[:,0:num_parties+1].sum(axis=1)

    vote_share_df = pd.DataFrame(index=election_df.index,columns=election_df.columns)
    for i in range(1,num_parties+1):                    #range(1,k) loops through 1, 2, . . . ,k-1
        vote_share_df[i] = election_df[i]/total_votes
    
    #Below is from GEO code
    #Set parameters for GEO metric
    min_cvs = 0.5 #Vote share losing districts must reach to become competitive
    max_cvs = 0.55 #Vote share that winning districts cannot drop below
    
        
    #Build lists of neighbors
    #neighbors[i] will contain list of neighbors of district i
    districts=election_df.index.tolist()  #Get list of districts from election_df
    neighbors = dict()  #Initiate empty dictionary of neighbors
    for district in districts:
        n_index = all_edges_df[(all_edges_df[0] == district)][1].tolist()   #Get index in all_edges_df of neighbors of i
        neighbors[district] = n_index  #Add values to neighbors list
    
    #List to hold GEO scores
    geo_scores_list = []
    
    #Loop through all parties
    for party in range(1,num_parties+1):    
        geo_score = 0
        newly_competitive = []

        
        geo_df = pd.DataFrame(index=election_df.index, columns=['Original Vote Share', 'Vote Share','Avg Neighbor Vote Share','Votes to Share','Made Competitive','Total Votes Shared','Is Competitive'])
        geo_df['Original Vote Share'] = vote_share_df[party]
        geo_df['Vote Share'] = vote_share_df[party]
        geo_df['Made Competitive'].values[:] = 0
        geo_df['Is Competitive'].values[:] = False          
        geo_df['Total Votes Shared'].values[:] = 0          
        
      
        #Compute Avg Neighbor Vote Share 
        for district in districts:
            total_neighborhood_votes = geo_df.loc[neighbors[district],'Vote Share'].sum() + geo_df.at[district,'Vote Share']
            geo_df.at[district,'Avg Neighbor Vote Share'] = total_neighborhood_votes / (len(neighbors[district])+1)
            
        #Use standard deviation of A_i to adjust votes to share, allow possibility of different adjustments for winning and losing districts 
        stdev = geo_df['Avg Neighbor Vote Share'].std()
        win_adj = stdev     #A_i - win_adj for winning districts
        loss_adj = stdev    #A_i - loss_adj for losing districts
        
            
        for district in districts:
            avg_neigh_vs = geo_df.at[district,'Avg Neighbor Vote Share']
            if geo_df.at[district,'Vote Share'] > max_cvs:   #Winning district that we potentially allow to share votes
                geo_df.at[district,'Votes to Share'] = max(0, geo_df.at[district,'Vote Share'] - max(max_cvs,avg_neigh_vs-win_adj))
            elif geo_df.at[district,'Vote Share'] >= min_cvs:  #Winning district we do not allow to share votes
                geo_df.at[district,'Votes to Share'] = 0
                geo_df.at[district,'Is Competitive'] = True            
            else:                                   #Losing district
                geo_df.at[district,'Votes to Share'] = max(0, geo_df.at[district,'Vote Share']-(avg_neigh_vs-loss_adj))
            
        #Sort by 'Avg Neighbor Vote Share', then get stable_wins and losses in this order
        geo_df.sort_values(by='Avg Neighbor Vote Share', axis=0, ascending=False, inplace=True)    
        stable_win = geo_df.index[(geo_df['Vote Share']>max_cvs)].tolist()
        loss =  geo_df.index[(geo_df['Vote Share']<min_cvs)].tolist()
            
        #All the districts to consider for shifting votes
        stable_win_loss = stable_win + loss
        
        #Run through loss districts to see if can make competitive
        for j in loss:
            needs_to_be_competitive = min_cvs - geo_df.at[j,'Vote Share'] 
                
            #Find vote shares that can be transferred in from neighbors
            shareable_neighbors = list( set.intersection( set(neighbors[j]), set(stable_win_loss)))
            neighbors_votes_to_share = geo_df.loc[shareable_neighbors,'Votes to Share'].sum()
            
            if needs_to_be_competitive <= neighbors_votes_to_share:  #If there's enough vote shares from neighbors to change district to competitive
                    # Adjust j to be competitive and remove j from stable_win_loss list
                    geo_df.at[j,'Vote Share'] = min_cvs
                    geo_df.at[j,'Votes to Share'] = 0          
                    geo_df.at[j,'Is Competitive'] = True
                    geo_df.at[j,'Made Competitive'] = True
                    newly_competitive.append(j)
                    stable_win_loss.remove(j)
                    
                    # Loop through shareable_neighbors, reducing votes to share
                    #  Reduce by proportion of votes neighbors have to share
                    sharing_neighbors = []
                    for k in shareable_neighbors:
                        if geo_df.at[k, 'Votes to Share'] == 0:    #It didn't end up sharing anything, so we don't want to note it
                            continue
                        sharing_neighbors.append(k)
                        votes_shared = (geo_df.at[k,'Votes to Share']/neighbors_votes_to_share) * needs_to_be_competitive
                        geo_df.at[k,'Vote Share'] -=  votes_shared
                        geo_df.at[k,'Votes to Share'] -=  votes_shared 
                        geo_df.at[k,'Total Votes Shared'] += votes_shared
                        
                            
        
        #Count number of non-zero values in 'Made Competitive'        
        geo_score = geo_df['Made Competitive'].astype(bool).sum()
      
        #Add to returned list:
        geo_scores_list.append(geo_score)
    
    return geo_scores_list
