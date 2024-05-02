from taipy.gui import Gui, Markdown
from features.time_series.create_container import create_container
import taipy.gui.builder as tgb
import pandas as pd
import os
import numpy as np


path_file_small = os.path.join("data","time_series","household_power_consumption_small.txt")
path_file_big = os.path.join("data","time_series","household_power_consumption.txt")

time_series_md = create_container(
    path_file_list=[
        {
            "path":path_file_small,
            'row_max': 50
        },
        {
            "path":path_file_big,
            'row_max': 5
        },        
    ]
)