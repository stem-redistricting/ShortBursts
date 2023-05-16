#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:39:49 2023

@author: erv2 and all authors of https://github.com/vrdi/shortbursts-gingles/blob/main/state_experiments/gingleator.py
"""

from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.updaters import (cut_edges, election)
from gerrychain.proposals import recom, propose_random_flip
from functools import (partial, reduce)
import numpy as np
import pandas as pd
import random
from statistics import mean
from helpers import geo


def config_markov_chain(initial_part, iters=1000, epsilon=0.05, compactness=True, 
                        geo_constraint = True, eg_constraint = False, pop="TOT_POP", accept_func=None, election_name = "T16SEN"):
    ideal_population = np.nansum(list(initial_part["population"].values())) / len(initial_part)

    proposal = partial(recom,
                       pop_col=pop,
                       pop_target=ideal_population,
                       epsilon=epsilon,
                       node_repeats=1)

    if compactness:
        compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]),
                            2*len(initial_part["cut_edges"]))
        cs = [constraints.within_percent_of_ideal_population(initial_part, epsilon),
              compactness_bound] 
    else:
        cs = [constraints.within_percent_of_ideal_population(initial_part, epsilon)]
        
    if eg_constraint:
        # Note: WE WILL WANT TO FIX THE BELOW SO THAT THE ELECTION IS NOT HARD CODED
        eg_bound = constraints.Bounds(lambda p: [p[election_name].efficiency_gap()], (-0.08, 0.08))  #brackets turn it into a list so it's iterable
        cs.append(eg_bound)
    
    #NOTE: Right now this checks that the difference in geo scores is between -8 and 8!
    #NOTE: Right now the GEO code is written so that the election is hard coded.  We'll want to change this later.
    #Surely this should eventually change
    if geo_constraint:
        geo_bound = constraints.Bounds(lambda p: [geo(p, election_name)[0] - geo(p, election_name)[1]], (-2, 2))
        cs.append(geo_bound)

    if accept_func == None: accept_func = accept.always_accept
    is_valid = constraints.Validator(cs)  #added
    return MarkovChain(proposal=proposal, constraints = is_valid,
                       accept=accept_func, initial_state=initial_part,
                       total_steps=iters) 



class Gingleator:
    """
    Gingleator class

    This class represents a set of methods used to find plans with greater numbers
    of gingles districts.
    """

    def __init__(self, initial_partition, threshold=0.4, 
                 score_funct=None, target_perc_col=None, eg = None,
                 pop_col="TOTPOP", epsilon=0.05, tot_seats = None, election_name = "T15SEN"):
        self.part = initial_partition
        self.threshold = threshold
        self.score = self.num_opportunity_dists if score_funct == None else score_funct
        self.target_perc = target_perc_col
        self.pop_col = pop_col
        self.epsilon = epsilon
        self.election_name = election_name



    def init_target_perc_col(self,target_pop_col, other_pop_col,
                               target_perc_col):
        """
         init_target_perc_col takes the string corresponding to the target
         population column and the desired name of the target percent 
         column and updates the partition updaters accordingly
         If the target column corresponds to D or R, target_pop_col + other_pop_col = total
         Otherwise, other_pop_col = total
        """
        last_char = target_pop_col[-1]
        if last_char == "D" or last_char == "R":
            perc_up = {target_perc_col:
                   lambda part: {k: part[target_pop_col][k] / (part[target_pop_col][k]+part[other_pop_col][k])
                                 for k in part.parts.keys()}}
        else: 
            perc_up = {target_perc_col:
                   lambda part: {k: part[target_pop_col][k] / part[other_pop_col][k]
                                 for k in part.parts.keys()}}
        self.part.updaters.update(perc_up)
    
    def init_total_seats(self, seats):
        """
        

        Parameters
        ----------
        seats : the number of districts in the state

        Returns
        -------
        initializes the number of seats.  This is needed for the eg class method

        """
        self.seats = seats

    """
    Types of Markov Chains:
    The following methods are different strategies for searching for the maximal
    number of Gingles districts
    """

    def short_burst_run(self, num_bursts, num_steps, verbose=False,
                        maximize=True, tracking_fun=None): #checkpoint_file=None):
        max_part = (self.part, self.score(self.part, self.target_perc,
                    self.threshold)) 
        """
        short_burst_run: preforms a short burst run using the instance's score function.
                         Each burst starts at the best preforming plan of the previous
                         burst.  If there's a tie, the later observed one is selected.
        args:
            num_steps:  how many steps to run an unbiased markov chain for during each burst
            num_bursts: how many bursts to preform
            verbose:    flag - indicates whether to prints the burst number at the beginning of 
                               each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
            tracking_fun: Function to save information about each observed plan.
        """
        observed_num_ops = np.zeros((num_bursts, num_steps))

        for i in range(num_bursts):
            if verbose: print("*", end="", flush=True)
            chain = config_markov_chain(max_part[0], iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col)

            for j, part in enumerate(chain):
                part_score = self.score(part, self.target_perc, self.threshold)
                observed_num_ops[i][j] = part_score
                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part

                if tracking_fun != None: tracking_fun(part, i, j)

        return (max_part, observed_num_ops)


    def variable_len_short_burst(self, num_iters, stuck_buffer=10,
                                 maximize=True, verbose=False):
        """
        variable_len_short_burst: preforms a variable length short burst run using the instance's 
                                  score function. Each burst starts at the best preforming plan of 
                                  the previous burst.  If there's a tie, the later observed one is 
                                  selected.
        args:
            num_iters:      the total number of steps to take (aka plans to sample)
            stuck_buffer:   Factor specifying how long to tolerate no improvement, before increasing
                            the burst length.
            verbose:        flag - indicates whether to prints the burst number at the beginning 
                                    of each burst
            maximize:       flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.target_perc,
                        self.threshold))
        observed_num_ops = np.zeros(num_iters)
        time_stuck = 0
        burst_len = 2
        i = 0

        while(i < num_iters):
            if verbose: print("*", end="", flush=True)
            chain = config_markov_chain(max_part[0], iters=burst_len,
                                        epsilon=self.epsilon, pop=self.pop_col)
            for j, part in enumerate(chain):
                part_score = self.score(part, self.target_perc, self.threshold)
                observed_num_ops[i] = part_score

                if part_score <= max_part[1]: time_stuck += 1
                else: time_stuck = 0

                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part
                
                i += 1
                if i >= num_iters: break
            if time_stuck >= stuck_buffer*burst_len : burst_len *= 2

        return (max_part, observed_num_ops)


    def biased_run(self, num_iters, p=0.25, maximize=True, verbose=False):
        """
        biased_run: preforms a biased (or tilted) run using the instance's score function.  The
                    chain always accepts a new proposal with the same or a better score and accepts
                    proposals with a worse score with some probability.
        args:
            num_iters:  total number of steps to take (aka plans to sample)
            p:          probability of a plan with a worse preforming score
            verbose:    flag - indicates whether to prints the burst number at the beginning 
                                    of each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.target_p,
                    self.threshold))
        observed_num_ops = np.zeros(num_iters)
        
        def biased_acceptance_function(part):
            if part.parent == None: return True
            part_score = self.score(part, self.target_perc, self.threshold)
            prev_score = self.score(part.parent, self.target_perc, self.threshold)
            if maximize and part_score >= prev_score: return True
            elif not maximize and part_score <= prev_score: return True
            else: return random.random() < p

        chain = config_markov_chain(self.part, iters=num_iters,
                                    epsilon=self.epsilon, pop=self.pop_col,
                                    accept_func= biased_acceptance_function)
        for i, part in enumerate(chain):
            if verbose and i % 100 == 0: print("*", end="", flush=True)
            part_score = self.score(part, self.target_perc, self.threshold)
            observed_num_ops[i] = part_score
            if maximize:
                max_part = (part, part_score) if part_score >= max_part[1] else max_part
            else:
                max_part = (part, part_score) if part_score <= max_part[1] else max_part

        return (max_part, observed_num_ops)


    def biased_short_burst_run(self, num_bursts, num_steps, p=0.25, 
                              verbose=False, maximize=True):
        """
        biased_short_burst_run: preforms a biased short burst run using the instance's score function.
                                Each burst is a biased run markov chain, starting at the best preforming 
                                plan of the previous burst.  If there's a tie, the later observed 
                                one is selected.
        args:
            num_steps:  how many steps to run an unbiased markov chain for during each burst
            num_bursts: how many bursts to preform
            p:          probability of a plan with a worse preforming score, within a burst
            verbose:    flag - indicates whether to prints the burst number at the beginning of 
                               each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.target_perc,
                    self.threshold)) 
        observed_num_ops = np.zeros((num_bursts, num_steps))

        def biased_acceptance_function(part):
            if part.parent == None: return True
            part_score = self.score(part, self.target_perc, self.threshold)
            prev_score = self.score(part.parent, self.target_perc, self.threshold)
            if maximize and part_score >= prev_score: return True
            elif not maximize and part_score <= prev_score: return True
            else: return random.random() < p

        for i in range(num_bursts):
            if verbose: print("Burst:", i)
            chain = config_markov_chain(max_part[0], iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col,
                                        accept_func= biased_acceptance_function)

            for j, part in enumerate(chain):
                part_score = self.score(part, self.target_perc, self.threshold)
                observed_num_ops[i][j] = part_score
                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part
    
        return (max_part, observed_num_ops)
    
    """
    Ellen added the EG run below
    """
    
    def eg_biased_short_burst_run(self, num_bursts, num_steps, p=0.25, 
                              verbose=False, maximize=True):
        """
        biased_short_burst_run: preforms a biased short burst run using the instance's score function.
                                Each burst is a biased run markov chain, starting at the best preforming 
                                plan of the previous burst.  If there's a tie, the later observed 
                                one is selected.
        args:
            num_steps:  how many steps to run an unbiased markov chain for during each burst
            num_bursts: how many bursts to preform
            p:          probability of a plan with a worse preforming score (|EG|>0.8) within a burst
            verbose:    flag - indicates whether to prints the burst number at the beginning of 
                               each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.target_perc,
                    self.threshold)) 
        observed_num_ops = np.zeros((num_bursts, num_steps))

        def biased_acceptance_function(part):
            if part.parent == None: return True
            part_score = self.score(part, self.target_perc, self.threshold)
            prev_score = self.score(part.parent, self.target_perc, self.threshold)
            eg_score = self.eg(part, self.target_perc, self.seats)
            #print("eg is", eg_score)
            if maximize and part_score >= prev_score and eg_score < 0.08 and eg_score > -0.08: return True
            elif not maximize and part_score <= prev_score  and eg_score < 0.08 and eg_score > -0.08: return True
            else: return random.random() < p

        for i in range(num_bursts):
            if verbose: print("Burst:", i)
            chain = config_markov_chain(max_part[0], iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col,
                                        accept_func= biased_acceptance_function)

            for j, part in enumerate(chain):
                part_score = self.score(part, self.target_perc, self.threshold)
                observed_num_ops[i][j] = part_score
                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part
    
        return (max_part, observed_num_ops)
    
    
    """
    Ellen added the GEO run below
    """
    
    def geo_biased_short_burst_run(self, num_bursts, num_steps, p=0.25, 
                              verbose=False, maximize=True):
        """
        biased_short_burst_run: preforms a biased short burst run using the instance's score function.
                                Each burst is a biased run markov chain, starting at the best preforming 
                                plan of the previous burst.  If there's a tie, the later observed 
                                one is selected.
        args:
            num_steps:  how many steps to run an unbiased markov chain for during each burst
            num_bursts: how many bursts to preform
            p:          probability of a plan with a worse preforming score (|EG|>0.8) within a burst
            verbose:    flag - indicates whether to prints the burst number at the beginning of 
                               each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.target_perc,
                    self.threshold)) 
        observed_num_ops = np.zeros((num_bursts, num_steps))

        def biased_acceptance_function(part):
            if part.parent == None: return True
            part_score = self.score(part, self.target_perc, self.threshold)
            prev_score = self.score(part.parent, self.target_perc, self.threshold)
            geo_scores = geo(part, self.election_name)
            print("geo is", geo_scores)
            if maximize and part_score >= prev_score and abs(geo_scores[0]-geo_scores[1])/self.seats < 1/5: return True #NOTE: This choice is super arbitrary!!!  
            # We need to discuss!  Same goes for below!!!
            elif not maximize and part_score <= prev_score and abs(geo_scores[0]-geo_scores[1])/self.seats < 1/5: return True
            else: return random.random() < p

        for i in range(num_bursts):
            if verbose: print("Burst:", i)
            chain = config_markov_chain(max_part[0], iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col,
                                        accept_func= biased_acceptance_function)

            for j, part in enumerate(chain):
                part_score = self.score(part, self.target_perc, self.threshold)
                observed_num_ops[i][j] = part_score
                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part
    
        return (max_part, observed_num_ops)


    """
    Score Functions
    """
    
    #Ellen added eg score below
    @classmethod
    def eg(cls, part, target_perc, seats):
        """
        eg: given a partition, name of the target percent updater, and the number of seats
                                that party won, return the Efficiency Gap (defined using only
                                seat share and vote share)
        """

        V = mean(part[target_perc].values())
        S = cls.num_opportunity_dists(part, target_perc, 0.5)/seats
        print("V is ", V, "S is", S, "eg is ", S-2*V+1/2)
        return S-2*V+1/2
    
        
        
    @classmethod
    def num_opportunity_dists(cls, part, target_perc, threshold):
        """
        num_opportunity_dists: given a partition, name of the target percent updater, and a
                               threshold, returns the number of opportunity districts.
        """
        dist_percs = part[target_perc].values()
        return sum(list(map(lambda v: v >= threshold, dist_percs)))


    @classmethod
    def reward_partial_dist(cls, part, target_perc, threshold):
        """
        reward_partial_dist: given a partition, name of the target percent updater, and a
                             threshold, returns the number of opportunity districts + the 
                             percentage of the next highest district.
        """
        dist_percs = part[target_perc].values()
        num_opport_dists = sum(list(map(lambda v: v >= threshold, dist_percs)))
        next_dist = max(i for i in dist_percs if i < threshold)
        return num_opport_dists + next_dist


    @classmethod
    def reward_next_highest_close(cls, part, target_perc, threshold):
        """
        reward_next_highest_close: given a partition, name of the target percent updater, and a
                                   threshold, returns the number of opportunity districts, if no 
                                   additional district is within 10% of reaching the threshold.  If one is, 
                                   the distance that district is from the threshold is scaled between 0 
                                   and 1 and added to the count of opportunity districts.
        """
        dist_precs = part[target_perc].values()
        num_opport_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        next_dist = max(i for i in dist_precs if i < threshold)

        if next_dist < threshold - 0.1:
            return num_opport_dists
        else: 
            return num_opport_dists + (next_dist - threshold + 0.1)*10


    @classmethod
    def penalize_maximum_over(cls, part, target_perc, threshold):
        """
        penalize_maximum_over: given a partition, name of the target percent updater, and a
                               threshold, returns the number of opportunity districts + 
                               (1 - the maximum excess) scaled to between 0 and 1.
        """
        dist_precs = part[target_perc].values()
        num_opportunity_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        if num_opportunity_dists == 0:
            return 0
        else:
            max_dist = max(dist_precs)
            return num_opportunity_dists + (1 - max_dist)/(1 - threshold)


    @classmethod
    def penalize_avg_over(cls, part, target_perc, threshold):
        """
        penalize_maximum_over: given a partition, name of the target percent updater, and a
                               threshold, returns the number of opportunity districts + 
                               (1 - the average excess) scaled to between 0 and 1.
        """
        dist_precs = part[target_perc].values()
        opport_dists = list(filter(lambda v: v >= threshold, dist_precs))
        if opport_dists == []:
            return 0
        else:
            num_opportunity_dists = len(opport_dists)
            avg_opportunity_dist = np.mean(opport_dists)
            return num_opportunity_dists + (1 - avg_opportunity_dist)/(1 - threshold)