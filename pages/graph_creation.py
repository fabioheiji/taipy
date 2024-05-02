from taipy.gui import Markdown
from features.graph.graph import create_graph, render_pydot, view_pydot
import pandas as pd
import networkx as nx
import io
import matplotlib.image as mpimg
from pathlib import Path
import os
from IPython.display import Image, display

file_path = os.path.join(Path(os.getcwd()), "data", "graph", "data.xlsx")
df = pd.read_excel(file_path)

# Create an empty graph
color = {
    'GCP':'red',
    'Uploader':'blue',
    'Script':'green'
}
G = create_graph(df, origem='origem', destino='destino', create_using_on=False, color=color, column_color='source')

S = [G.subgraph(c).copy() for c in nx.connected_components(G)]  

img_list = []
img_list_markdown = ''
for idx, sG in enumerate(S):
    nodes_subgraph = sG.nodes()
    df_subgraph = df.query('origem in @nodes_subgraph or destino in @nodes_subgraph')

    G = create_graph(df_subgraph, color=color, column_color='source')
    render_pydot(G)
    graph = nx.drawing.nx_pydot.to_pydot(sG)
    plt = Image(graph.create_png())
    display(plt)    
    view_pydot(graph)
    # Render the pydot graph by calling 'dot' (GraphViz) without saving any files
    png_str = graph.create_png(prog='dot')
    img_list.append(png_str)
    img_list_markdown += '<|{img_list[*]}|image|class_name=--container-max-width|>'.replace('*', str(idx))
       

    # break



graph_creation_md = Markdown(
"""
# Module *Graph Creation*

"""
+
img_list_markdown
)
# <|{png_str}|image|>
# <|img_list[1]|image|>
# <|{img_list}|image|>
# <|{img}|image|label=This is an image|on_action=function_name|>

# def function_name(state):
#     print('foi')