import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Function for extractive summarization using sumy
def extractive_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    summary_text = " ".join([str(sentence) for sentence in summary])
    return summary_text

# Function for abstractive summarization using transformers
def abstractive_summary(text, max_length=150):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
def main():
    st.title("Text Summarization App")
    st.write("Enter text and choose whether you want an extractive or abstractive summary.")

    text = st.text_area("Input Text", height=300)
    summary_type = st.selectbox("Summary Type", ["Extractive", "Abstractive"])

    if st.button("Summarize"):
        if summary_type == "Extractive":
            summary = extractive_summary(text)
        elif summary_type == "Abstractive":
            summary = abstractive_summary(text)
        st.write("## Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
