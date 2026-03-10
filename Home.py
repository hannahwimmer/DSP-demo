import streamlit as st

st.set_page_config(page_title="Digital Signal Processing", layout="wide")

st.title("Methods and Topics of DSP - Overview")
st.markdown("""
Here's a short demo on how you should document your findings on each topic we discuss
in class. In the end, you should have a nice package of demos implemented in Python, 
with a Streamlit app to visualize and compare different settings and their effect.  

Use the menu to switch between pages:
- **Short Streamlit Demo:** here's a short demo on what each of your home assignements
  should look like (no need for 100k words, just keep it short but informative)
- **To Be Continued:** here's an empty page for demonstrating the pages setup. Add a
  novel page for each assignment so that you can browse through your insights at the end
  of the term when preparing for the exam.
""")

st.badge("The page on each topic should include:", color="blue")
st.markdown("""
1. Introduction  
2. Methods  
3. Results  
4. Discussion
""")