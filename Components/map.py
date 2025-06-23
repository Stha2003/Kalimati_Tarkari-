import streamlit as st
import pandas as pd
import pydeck as pdk

def show_kalimati_map(df):
  
    st.subheader(" Kalimati Market Location")
    kalimati_location = pd.DataFrame({"lat": [27.6973], "lon": [85.3065]})
    st.map(kalimati_location)