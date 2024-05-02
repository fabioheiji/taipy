from taipy.gui import Gui, Markdown
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from typing import List, Dict

def create_container(path_file_list: List[Dict[str,str]]):
    
    def select_hist_colums(state, var_name, value):
        idx = var_name.split("_")[-1]
        # exec(f"state.df_hist_{idx} = state.df_{idx}[state.hist_columns_{idx}]")
        exec(f"state.df_hist_{idx} = [{{k:list(v)}} for k,v in state.df_{idx}[state.hist_columns_{idx}].to_dict(orient='series').items()]")
        exec(f"""
state.fig_hist_{idx} = go.Figure()
for idx_hist_column, hist_column in enumerate(state.hist_columns_{idx}):
    state.fig_hist_{idx}.add_trace(go.Histogram(name=hist_column, x=state.df_hist_{idx}[idx_hist_column][hist_column]))
        """
        )

        exec(f"state.fig_hist_{idx}.update_layout(barmode='overlay', legend=dict(xanchor='center', y=1.1, x=0.5, orientation='h'))")
        exec(f"state.fig_hist_{idx}.update_traces(opacity=0.75)")
    
    def select_boxplot_colums(state, var_name, value):
        idx = var_name.split("_")[-1]
        exec(f"state.df_boxplot_{idx} = state.df_{idx}[state.boxplot_columns_{idx}]")
        exec(f"""
state.fig_boxplot_{idx} = go.Figure()
for boxplot_column in state.boxplot_columns_{idx}:
    state.fig_boxplot_{idx}.add_trace(go.Violin(name=boxplot_column,x=state.df_boxplot_{idx}[boxplot_column].values, box_visible=True, meanline_visible=True))
        """
        )
        exec(f"state.fig_boxplot_{idx}.update_layout(legend=dict(xanchor='center', y=1.1, x=0.5, orientation='h'))")
    
    layout = {
        # Overlay the two histograms
        "barmode": "overlay",
        # Hide the legend
        "showlegend": False
    }
    
    raw_text = ''
    df_dict = {}
    hist_all_columns_dict = {}
    hist_columns_dict = {}
    df_hist_dict = {}
    
    boxplot_all_columns_dict = {}
    boxplot_columns_dict = {}
    df_boxplot_dict = {}
    for idx,path_file in enumerate(path_file_list):        

        df = pd.read_csv(path_file['path'], sep=";", nrows=path_file['row_max'])
        exec(f"df_{idx} = df")
        exec(f"df_dict[{idx}] = df_{idx}")
        
        # Histogram parameters
        exec(f"hist_all_columns_{idx} = list(df_{idx}.columns)")
        exec(f"hist_all_columns_dict[{idx}] = hist_all_columns_{idx}")        

        exec(f"hist_columns_{idx} = ['Global_active_power']")
        exec(f"hist_columns_dict[{idx}] = hist_columns_{idx}")

        exec(f"df_hist_{idx} = [{{k:list(v)}} for k,v in df_{idx}[hist_columns_{idx}].to_dict(orient='series').items()]")

        exec(f"df_hist_dict[{idx}] = df_hist_{idx}")

        # Boxplot parameters
        exec(f"boxplot_all_columns_{idx} = list(df_{idx}.columns)")
        exec(f"boxplot_all_columns_dict[{idx}] = boxplot_all_columns_{idx}")        

        exec(f"boxplot_columns_{idx} = ['Global_active_power']")
        exec(f"boxplot_columns_dict[{idx}] = boxplot_columns_{idx}")

        exec(f"df_boxplot_{idx} = df_{idx}[boxplot_columns_{idx}]")
        exec(f"df_boxplot_dict[{idx}] = df_boxplot_{idx}")

        exec(f"""
fig_boxplot_{idx} = go.Figure()
for boxplot_column in boxplot_columns_{idx}:
    fig_boxplot_{idx}.add_trace(go.Violin(name=boxplot_column, x=df_boxplot_{idx}[boxplot_column].values, box_visible=True, meanline_visible=True))
            """
            )
        exec(f"fig_boxplot_{idx}.update_layout(legend=dict(xanchor='center', y=1.1, x=0.5, orientation='h'))")


        exec(f"""
fig_hist_{idx} = go.Figure()
for idx_hist_column, hist_column in enumerate(hist_columns_{idx}):
    fig_hist_{idx}.add_trace(go.Histogram(name=hist_column, x=df_hist_{idx}[idx_hist_column][hist_column]))
            """
            )

        exec(f"fig_hist_{idx}.update_layout(barmode='overlay', legend=dict(xanchor='center', y=1.1, x=0.5, orientation='h'))")
        exec(f"fig_hist_{idx}.update_traces(opacity=0.75)")

        raw_text += """
<|Endpoint|expandable|expanded=False|class_name=mb1|

<|{df_*}|table|rebuild|height=40vh|>


<|layout|columns=1 1|

<|
<center><|{hist_columns_*}|selector|multiple=True|id=*|dropdown=True|propagate=True|lov={hist_all_columns_*}|on_change=select_hist_colums|></center>
<center><|chart|figure={fig_hist_*}|></center>
|>

<|
<center><|{boxplot_columns_*}|selector|multiple=True|id=*|dropdown=True|propagate=True|lov={boxplot_all_columns_*}|on_change=select_boxplot_colums|></center>
<center><|chart|figure={fig_boxplot_*}|></center>
|>

|>


|>

""".replace('*',str(idx))
    
    # <center><|{df_hist_*}|chart|rebuild|type=histogram|layout={layout}|></center>

    return Markdown(raw_text)