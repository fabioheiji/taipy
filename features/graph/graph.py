import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import pydot
from graphviz import Source
from IPython.display import Image, display

def create_graph(df:pd.DataFrame, origem='origem', destino='destino', create_using_on=True, color:dict=None, column_color:str=None):
    if create_using_on:
        G = nx.from_pandas_edgelist(df, origem, destino, create_using=nx.DiGraph)
    else:
        G = nx.from_pandas_edgelist(df, origem, destino)
    # Compute node positions using spring layout
    pos = nx.kamada_kawai_layout(G)

    # Add the pos attribute to the nodes
    nx.set_node_attributes(G, pos, 'pos')
    if color is not None:
        for node in list(df['origem']):
            criteria_color = df.query('origem == @node')[column_color].unique()[0]
            
            if criteria_color in color:
                G.nodes[node]['color'] = color[criteria_color]
            else:
                G.nodes[node]['color'] = 'white'
            G.nodes[node]['style'] = 'filled'

        nx.set_node_attributes(G, color, 'color')

    return G

def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

def render_pydot(sG):
    # for node_name in list(sG.nodes()):
    #     sG.nodes[node_name]['style'] = 'filled'
    #     sG.nodes[node_name]['color'] = 'red'

    graph = nx.drawing.nx_pydot.to_pydot(sG)
    view_pydot(graph)  