import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import collections
from IPython.display import Image

COUNTRIES = ["USA", "China", "United Kingdom", "Australia"]
COORDINATES = {"USA": [-179, -66, 16, 60],
               "China": [70, 140, 15, 53],
               "United Kingdom": [-11, 2, 48, 60],
               "Australia": [111, 155, -42, -9]}

LBL_W = {"USA": 213000,             # 50
         "China": 51100,            # 75
         "United Kingdom": 48700,   # 17
         "Australia": 10704}           # 20

WIDTH_C = {"USA": 0.5,
           "China": 0.75,
           "United Kingdom": 0.9,
           "Australia": 5}

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

# makes sure weight is int
routes_2003["Weight"] = routes_2003["Weight"].astype(int)
routes_2016["Weight"] = routes_2016["Weight"].astype(int)

# ignoring time series as 2016 doesn't have it
routes_2003 = routes_2003.drop(["TimeSeries"], axis=1)
# instead add a columns to differentiate between old (2003-2009) and new (2010-2016)
routes_2003["isOld"] = 1
routes_2016["isOld"] = 0

# join the two routes in one df
routes_tot = routes_2003.append(routes_2016, ignore_index=True)


def network_graph(country, airports_names=airports_names, routes=routes_tot, plot=True, output=False):
    country = "United Kingdom" if country.upper() == "UK" else country
    airport_country_filter = "United States" if country.upper() == "USA" else country
    airports_country = airports_names[airports_names["country"] == airport_country_filter]

    if country.upper() == "USA":
        airports_country = airports_country[airports_country["Lon"] != -70]

    routes_country = routes[routes["Source Country"] == country]
    routes_country = routes_country[routes_country["Source"].isin(airports_country.index) &
                                    routes_country["Target"].isin(airports_country.index)]

    weight_edges = routes_country[["Source", "Target", "Weight"]].values
    g = nx.Graph()
    g.add_weighted_edges_from(weight_edges)

    pos = {airport: (v["Lon"], v["Lat "]) for airport, v in
           airports_country.to_dict('index').items()}

    deg = nx.degree(g, weight='weight')
    all_sizes = [deg[iata] for iata in g.nodes]
    sizes = [(((deg[iata] - min(all_sizes)) * (300 - 17)) / (max(all_sizes) - min(all_sizes))) + 1 for iata in g.nodes]

    labels = {iata: iata if deg[iata] >= LBL_W[country] else ''
              for iata in g.nodes}

    all_weights = [data['weight'] for node1, node2, data in g.edges(data=True)]
    edge_width = [(((weight - min(all_weights)) * (WIDTH_C[country] - 0.075)) / (max(all_weights) - min(all_weights))) + 0.075
                  for weight in all_weights]

    if plot:
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            1, 1, figsize=(17, 8),
            subplot_kw=dict(projection=crs))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        # Extent of continental US.
        ax.set_extent(COORDINATES[country])
        ax.gridlines()
        nx.draw_networkx(g, ax=ax,
                         font_size=17,
                         alpha=.5,
                         width=edge_width,
                         node_size=sizes,
                         labels=labels,
                         pos=pos,
                         node_color=sizes,
                         cmap=plt.cm.plasma)
        plt.show()

    if output:
        return airports_country, routes_country, g, weight_edges, pos, deg, sizes, labels, all_weights, edge_width


# plot f degree distribution
def degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())


# degree vs betweenness distr
def degree_betweenness(G):
    pass


# assortativity
def assort(G):
    pass


# core community size
def core_community(G):
    pass


if __name__ == "__main__":
    network_graph("UK")
