#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:13:02 2024

@author: eveomett
"""

import random
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

seed_location_prefix = "./data/seeds/"

states_and_elections = {("AK_permissive/AK_precincts/alaska_precincts.shp", "PRES16", "AK"),
                        ("MA_precincts_12_16/MA_precincts_12_16.shp", "SEN18", "MA"),
                        ("MI/mi16_results.shp", "PRES16", "MI"),
                        ("OK_precincts/OK_precincts.shp", "GOV18", "OK"),
                        ("OR_precincts/OR_precincts.shp", "GOV18", "OR"),
                        ("PA/PAnew/PA.shp", "T16SEN", "PA"),
                        ("TX_vtds/TX_vtds.shp", "SEN14", "TX")
                        }
for (state, election, statename) in states_and_elections:
    fp = seed_location_prefix + state
    
    
    map_df = gpd.read_file(fp)

    print(type(map_df))
    
    map_df.plot()
    
    df=gpd.read_file(seed_location_prefix + state)

    print(df.columns)
    df["Partisan Lean"] = df[election + "R"]/(df[election + "R"]+df[election + "D"])
    print(df["Partisan Lean"])
    
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axis('off')
    ax.set_title('Partisan lean', fontdict={'fontsize': '15', 'fontweight' : '3'})
    colorscale = ["rgb(255, 51, 51)", "rgb(48, 48, 255)"]
    df.plot(column='Partisan Lean',
                #cmap = "RdYlBu",
                cmap = 'coolwarm',
                #cmap='bwr',
                linewidth=0.05,
                ax=ax,
                edgecolor='0.4',
                vmin = 0,
                vmax = 1,
                legend=True, missing_kwds={
                "color": "lightgrey",
                "label": "Missing values",},)
    plt.savefig('./data/choropleths/' + statename + '.png' )