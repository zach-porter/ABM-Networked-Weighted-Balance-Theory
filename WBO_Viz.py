#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import mesa
import asyncio
import matplotlib
import numpy as np
from WBO_opinion_mod import WeightedBalanceModel


# In[2]:


def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    portrayal = {}
    portrayal["nodes"] = [
        {
            "id": node_id,
            "size": 20,
            "color": get_color_for_opinion(agents[0].opinion_values),
            "label": None,
        }
        for (node_id, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {"id": edge_id, "source": source, "target": target, "color": "#000000"}
        for edge_id, (source, target) in enumerate(G.edges)
    ]

    return portrayal

def get_color_for_opinion(opinion_value, n = 0):
    # opinion values are between -1 and 1
    normalized_opinion = (opinion_value + 1) / 2  # Normalize to [0, 1]
    color = matplotlib.cm.plasma(normalized_opinion)
    hex_color = matplotlib.colors.rgb2hex(color[0])  # Use [0] to extract the first color
    return hex_color

def network_portrayal1(G):
    # The model ensures there is always 1 agent per node

    portrayal1 = {}
    portrayal1["nodes"] = [
        {
            "id": node_id,
            "size": 20,
            "color": get_color_for_opinion1(agents[0].opinion_values),
            "label": None,
        }
        for (node_id, agents) in G.nodes.data("agent")
    ]

    portrayal1["edges"] = [
        {"id": edge_id, "source": source, "target": target, "color": "#000000"}
        for edge_id, (source, target) in enumerate(G.edges)
    ]

    return portrayal1

def get_color_for_opinion1(opinion_value, n = 0):
    # opinion values are between -1 and 1
    normalized_opinion1 = (opinion_value + 1) / 2  # Normalize to [0, 1]
    color1 = matplotlib.cm.plasma(normalized_opinion1)
    hex_color1 = matplotlib.colors.rgb2hex(color1[1])  # Use [1] to extract the second opinion
    return hex_color1

def network_portrayal2(G):
    # The model ensures there is always 1 agent per node

    portrayal2 = {}
    portrayal2["nodes"] = [
        {
            "id": node_id,
            "size": 20,
            "color": get_color_for_opinion2(agents[0].opinion_values),
            "label": None,
        }
        for (node_id, agents) in G.nodes.data("agent")
    ]

    portrayal2["edges"] = [
        {"id": edge_id, "source": source, "target": target, "color": "#000000"}
        for edge_id, (source, target) in enumerate(G.edges)
    ]

    return portrayal2

def get_color_for_opinion2(opinion_value, n = 0):
    # opinion values are between -1 and 1
    normalized_opinion2 = (opinion_value + 1) / 2  # Normalize to [0, 1]
    color2 = matplotlib.cm.plasma(normalized_opinion2)
    hex_color2 = matplotlib.colors.rgb2hex(color2[2])  # Use [2] to extract the third opinion
    return hex_color2



network = mesa.visualization.NetworkModule(network_portrayal, 250, 250)

network1 = mesa.visualization.NetworkModule(network_portrayal1, 250, 250)

network2 = mesa.visualization.NetworkModule(network_portrayal2, 250, 250)

#agent3d = mesa.visualization.ChartModule(opinion_portrayal, 500, 500)

chart = mesa.visualization.ChartModule(
    [{"Label": "Polarization", "Color": "Black"}, 
     {"Label": "Alignment", "Color": "Blue"},
     {"Label": "Extremeness", "Color": "Red"}],
     data_collector_name = "datacollector")



model_params = {
    "n_agents": mesa.visualization.Slider(
        "Number of agents",
        7,
        2,
        500,
        1,
        description="Choose how many agents to include in the model",
    ),
    "n_opinions": mesa.visualization.Slider(
        "Number of opinions",
        3,
        1,
        10,
        1,
        description="Choose how many opinions each agent has",
    ),
    "network": mesa.visualization.Choice(
        'My network',
        value = 'None',
        choices = ['None' ,'Wattz-Strogatz', 'Connected Wattz-Strogatz']),
    "threshold": mesa.visualization.Slider(
        "Threshold of Burt Score value",
        .5,
        0,
        1,
        .05,
        description="Set the threshold value of the Burt Score",),
    "radius": mesa.visualization.Slider(
        "radius/distance of the neighbor for interaction",
        1,
        1,
        10,
        1,
        description="Set the threshold value of the Burt Score",),
     "mean": mesa.visualization.Slider(
        "mean of epsilon value",
        .5,
        0,
        1,
        .1,
        description="Choose the mean of the epsilon",
    ),
     "std": mesa.visualization.Slider(
        "std of epsilon",
        .5,
        0,
        1,
        .1,
        description="Choose std of epsilon",
    ),
}

server = mesa.visualization.ModularServer(
    WeightedBalanceModel, [network, network1, network2, chart], "Opinion Model", model_params
)
server.port = 8521

server.launch()


# In[ ]:





# In[ ]:





# In[ ]:




