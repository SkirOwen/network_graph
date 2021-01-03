import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from IPython.display import Image

COUNTRIES = ["USA", "China", "United Kingdom", "Australia"]
COORDINATES = {"USA": [-179, -66, 10, 55],
               "China": [70, 140, 15, 53],
               "United Kingdom": [-11, 2, 48, 60],
               "Australia": [111, 155, -42, -9]}

# Importing the data
airports_names = pd.read_csv("Airports.csv")
routes_2003 = pd.read_csv("2003-2009.csv")
routes_2016 = pd.read_csv("2010-2016.csv", dtype=object)

# Cleaning the data
# set IATA/labels as index of df
airports_names = airports_names.set_index("id")

# remove data for countries not USA, China, UK, and Australia
routes_2003 = routes_2003[routes_2003["Source Country"].isin(COUNTRIES)].reset_index(drop=True)
routes_2016 = routes_2016[routes_2016["Source Country"].isin(COUNTRIES)].reset_index(drop=True)

# ignoring time series as 2016 doesn't have it
routes_2003 = routes_2003.drop(["TimeSeries"], axis=1)

# join the two routes in one df
routes_tot = routes_2003.append(routes_2016, ignore_index=True)


def network_graph(country, airports_names=airports_names, routes=routes_tot):
    airport_country_filter = "United States" if country.upper() == "USA" else country
    airports_country = airports_names[airports_names["country"] == airport_country_filter]

    if country.upper() == "USA":
        airports_country = airports_country[airports_country["Lon"] != -70]

    routes_country = routes[routes["Source Country"] == country]
    routes_country = routes_country[routes_country["Source"].isin(airports_country.index) &
                                    routes_country["Target"].isin(airports_country.index)]

    edges = routes_country[["Source", "Target"]].values
    g = nx.from_edgelist(edges)

    pos = {airport: (v["Lon"], v["Lat "]) for airport, v in
           airports_country.to_dict('index').items()}

    deg = nx.degree(g)
    sizes = [2 * deg[iata] for iata in g.nodes]

    labels = {iata: iata if deg[iata] >= 32 else ''
              for iata in g.nodes}

    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 8),
        subplot_kw=dict(projection=crs))
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    # Extent of continental US.
    ax.set_extent(COORDINATES[country])
    ax.gridlines()
    nx.draw_networkx(g, ax=ax,
                     font_size=15,
                     alpha=.5,
                     width=.075,
                     node_size=sizes,
                     labels=labels,
                     pos=pos,
                     node_color=sizes,
                     cmap=plt.cm.plasma)
    plt.show()


airports_us = airports_names[airports_names["country"] == "United States"]
airports_us = airports_us[airports_us["Lon"] != -70]
routes_us = routes_2003[routes_2003["Source Country"] == "USA"]
routes_us = routes_us.append(routes_2016[routes_2016["Source Country"] == "USA"], ignore_index=True)

#       Geographic visualisation

# Remove missing airports from the routes
routes_us = routes_us[routes_us["Source"].isin(airports_us.index)]
routes_us = routes_us[routes_us["Target"].isin(airports_us.index)]

edges = routes_us[["Source", "Target"]].values
g = nx.from_edgelist(edges)

pos = {airport: (v["Lon"], v["Lat "]) for airport, v in
       airports_us.to_dict('index').items()}

deg = nx.degree(g)
sizes = [2 * deg[iata] for iata in g.nodes]

labels = {iata: iata if deg[iata] >= 32 else ''
          for iata in g.nodes}

crs = ccrs.PlateCarree()
fig, ax = plt.subplots(
    1, 1, figsize=(12, 8),
    subplot_kw=dict(projection=crs))
ax.coastlines()
# Extent of continental US.
ax.set_extent([-179, -66, 15, 59])
ax.gridlines()
nx.draw_networkx(g, ax=ax,
                 font_size=15,
                 alpha=.5,
                 width=.075,
                 node_size=sizes,
                 labels=labels,
                 pos=pos,
                 node_color=sizes,
                 cmap=plt.cm.plasma)
plt.show()

## plot f degree distribution

## degree vs betweenness distr

## assortativity

## core community size
