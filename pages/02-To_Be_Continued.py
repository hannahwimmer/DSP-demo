import base64
import streamlit as st


st.title("Here's where your home assignments are gonna end up...")

st.markdown("""
For each new topic, add a novel page to the Streamlit setup. Use the current demo
as a template and adapt everything however you see fit. Looking forward to seeing your
work!
""")

gif = open("docs/images/fran.gif", "rb")
contents = gif.read()
data_url = base64.b64encode(contents).decode("utf-8")
gif.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}">',
    unsafe_allow_html=True,
)