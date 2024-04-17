import torch
import wikipedia
import transformers
import streamlit as st
from transformers import pipeline, Pipeline

def load_qa_pipeline() -> Pipeline :
    qa_pipeline = pipeline("question-answering", model = "distilbert-base-uncased-distilled-squad")
    if not qa_pipeline:
        return "No answer found for this question."
    return qa_pipeline

def load_wiki_summary(query: str) -> str:
    results = wikipedia.search(query)
    # Check if search results are empty
    if not results:
        return "No Wikipedia page found for the given topic."
    summary = wikipedia.summary(results[0], sentences = 10)
    return summary

def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> dict:
    input = {
        "question" : question,
        "context" : paragraph
    }
    output = pipeline(input)
    return output

def main():
    # display title and description
    st.title('Wikipedia Question Answering 📜')
    st.write("Search topic, Ask questions, Get answers")
    st.write('\n')

    # display topic input slot
    topic = st.text_input("Search Topic :", "")

    # display article paragraph
    article_paragraph = st.empty()

    if topic:
        # display processing animation while loading
        with st.spinner("Loading Wikipedia summary..."):
            # load wikipedia summary of topic
            summary = load_wiki_summary(topic)

            # display article summary in paragraph
            article_paragraph.markdown(summary)

    # display question input slot
    question =  st.text_input("Question :", "")

    with st.spinner("Loading answer ..."):
        if question != "":
            # load question answer
            qa_pipeline = load_qa_pipeline()

            # answer query question using article summary
            result = answer_question(qa_pipeline, question, summary)
            answer = result['answer']

            # display answer 
            st.write(answer)       
    
    

# Main app engine
if __name__ == '__main__':
    main()