import streamlit as st

st.set_page_config(layout="wide", page_title="Fusion Geometry Visualizer")

# Define pages for navigation
PAGES = {
    "Visualization": [
        st.Page("pages/reactor-visualisation.py", title="Reactor Visualizer"),
        st.Page("pages/network-visualisation.py", title="Network Visualizer"),
    ],
}

# Run the navigation
if __name__ == "__main__":
    pages = st.navigation(pages=PAGES, position="top")
    pages.run()
