import streamlit as st

# Define the pages of the app
home_page = st.Page("pages/app.py", title="Summarize your PDFs")
about_page = st.Page("pages/about.py", title="About")

# Set up the navigation
navigation = st.navigation([home_page, about_page])

# Run the selected page
navigation.run()
