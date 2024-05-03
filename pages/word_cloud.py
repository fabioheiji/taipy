from taipy.gui import Markdown
import random
from features.word_cloud.generate_plotly import plotly_wordcloud
import pandas as pd

word_list = random.choices('Chaotic college protests and stalled effort to bring about Gaza cease-fire leave president with few good options and plenty of risks'.split(' '), k=1000)

fig_wordcloud = plotly_wordcloud(' '.join(word_list))

word_cloud_md = Markdown(
"""

<|testando|text|>

<|chart|figure={fig_wordcloud}|>

"""
)