from taipy.gui import Gui, navigate
from pages.graph_creation import graph_creation_md
from pages.llm import llm_md
from pages.home import home_md
from pages.time_series import time_series_md
from pages.word_cloud import word_cloud_md

pages = {
    "/": "<|menu|lov={page_names}|on_action=menu_action|>",
    "home": home_md,
    "graph_creation": graph_creation_md,
    "llm": llm_md,
    "time_series": time_series_md,
    'word': word_cloud_md,
}


page_names = [page for page in pages.keys() if page != "/"]

def menu_action(state, action, payload):
    page = payload["args"][0]
    navigate(state, page)

if __name__ == "__main__":
    Gui(pages=pages).run(title="Dynamic chart", port=5002, debug=True, use_reloader=True, run_browser=False)