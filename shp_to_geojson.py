#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:53:51 2023

@author: eveomett
"""

import geopandas

shp_file = geopandas.read_file('./data/seeds/OK_precincts/OK_precincts.shp')

shp_file.to_file('./data/seeds/OK_precincts/OK.geojson', driver='GeoJSON')
