import streamlit as st

# Define pages for navigation
PAGES = {
    "Visualization": [
        st.Page("pages/reactor-visualisation.py", title="Reactor Visualizer"),
        st.Page("pages/network-visualisation.py", title="Network Visualizer"),
        st.Page("pages/benchmark-visualisation.py", title="Benchmark Visualizer"),
    ],
}

# Run the navigation
if __name__ == "__main__":
    pages = st.navigation(pages=PAGES, position="top")
    pages.run()
