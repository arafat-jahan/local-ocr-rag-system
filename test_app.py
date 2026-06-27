import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")
st.title("✅ Streamlit Test App")
st.write("This is a simple test to make sure Streamlit works!")
st.write("If you're seeing this, everything is good!")

name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! 👋")
