import streamlit as st
from transformers import pipeline

# Load pre-trained model and tokenizer
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Streamlit app
st.title("Question Answering Chatbot")
st.write("Enter your question and context below:")

# User inputs
question = st.text_input("Question")
context = st.text_area("Context")

# Answer generation
if st.button("Get Answer"):
    if question and context:
        result = qa_pipeline(question=question, context=context)
        st.write("**Answer:**", result['answer'])
    else:
        st.write("Please provide both a question and a context.")
