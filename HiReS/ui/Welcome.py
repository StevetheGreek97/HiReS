import streamlit as st


st.set_page_config(
    page_title="HiReS – Run Pipeline",
    layout="wide",
)

st.title("Run HiReS Pipeline")

st.caption(
    "Configure the pipeline below, then click **Run HiReS pipeline** to execute "
    "chunk → predict → unify → NMS → crops → shape descriptors."
)

