import streamlit as st

st.title("Test App")
st.write("Hello, World!")

number = st.slider("Select a number", 0, 100, 50)
st.write(f"You selected: {number}")
