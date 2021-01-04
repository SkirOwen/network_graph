import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import collections
from IPython.display import Image
import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.style.use('seaborn-whitegrid')

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


def network_graph(country, airports_names=airports_names, routes=routes_tot, plot=True, output_g=False, output_all=False):
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

    if output_all:
        return airports_country, routes_country, g, weight_edges, pos, deg, sizes, labels, all_weights, edge_width
    if output_g:
        return g


# plot f degree distribution
def degree_distribution(deg, country):
    if len(deg) > 1:
        degree_sequence_old = sorted([d for n, d in deg[0]], reverse=True)
        degree_sequence_new = sorted([d for n, d in deg[1]], reverse=True)
        degree_sequence_all = sorted([d for n, d in deg[2]], reverse=True)

        plt.semilogy(degree_sequence_old, marker="o", label="2003-2009")
        plt.semilogy(degree_sequence_new, marker="o", label="2010-2016")
        plt.semilogy(degree_sequence_all, marker="o", label="2003-2016")
        plt.legend(loc="best")
    else:
        degree_sequence = sorted([d for n, d in deg[0]], reverse=True)
        plt.semilogy(degree_sequence, marker="o")

    plt.title(r'Degree Distribution' + " for " + str(country))
    plt.ylabel(r"Weighted Degree")
    plt.xlabel(r"Rank")
    plt.show()


# degree vs betweenness distr
def degree_betweenness(G, deg, country):
    # init the plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog()

    if len(deg) > 1:
        time_label = ["2003-2009", "2010-2016", "2003-2016"]

        for k in range(len(deg)):
            b = nx.betweenness_centrality(G[k], weight="weight", normalized=False)
            x = [deg[k][iata] for iata in G[k].nodes]
            y = [b[iata] for iata in G[k].nodes]
            labels = [iata for iata in G[k].nodes]

            plt.scatter(x, y, alpha=0.75, label=time_label[k])
            for i in range(len(labels)):
                ax.annotate(labels[i], (x[i], y[i]))
    else:
        b = nx.betweenness_centrality(G, weight="weight", normalized=False)
        x = [deg[iata] for iata in G.nodes]
        y = [b[iata] for iata in G.nodes]
        labels = [iata for iata in G.nodes]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog()
        plt.scatter(x, y, alpha=0.75)
        for i in range(len(labels)):
            ax.annotate(labels[i], (x[i], y[i]))

    plt.ylim(0.1, 10000)
    plt.legend(loc="best")
    plt.title(r'Degree vs Betweenness' + " for " + str(country))
    plt.ylabel(r"Betweenness")
    plt.xlabel(r"Weighted Degree")
    plt.show()

# assortativity
def assort(G):
    r = nx.degree_pearson_correlation_coefficient(G, weight="weight")
    return r


# core community size
def core_community(G):
    pass


if __name__ == "__main__":
    country = "China"   # this to change
    g_old = network_graph(country, routes=routes_2003, output_g=True)
    g_new = network_graph(country, routes=routes_2016, output_g=True)
    g_all = network_graph(country, output_g=True)

    deg = [nx.degree(g_old, weight='weight'), nx.degree(g_new, weight='weight'), nx.degree(g_all, weight='weight')]
    G = [g_old, g_new, g_all]

    # if deg and G is a list of old, new, and all than plot the three on one graph
    degree_betweenness(G, deg, country)

    degree_distribution(deg, country)

    r_old = assort(g_old)
    r_new = assort(g_new)
    r_all = assort(g_all)
    print(country + " & " + str(r_old) + " & " + str(r_new) + " & " + str(r_all) + "\\")
