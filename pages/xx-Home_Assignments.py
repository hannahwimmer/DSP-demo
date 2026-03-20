import base64
import streamlit as st


st.title("Structure your home assignments as Streamlit pages!")

st.markdown("""
For each new topic, add a novel page to the Streamlit setup. Use the current demo
as a template, remove my parts and adapt everything however you see fit. 

**Looking forward to seeing your work!**
""")

gif = open("docs/images/fran.gif", "rb")
contents = gif.read()
data_url = base64.b64encode(contents).decode("utf-8")
gif.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}">',
    unsafe_allow_html=True,
)


# https://www.youtube.com/shorts/8E36bKfg_qU --> add that somewhere...
