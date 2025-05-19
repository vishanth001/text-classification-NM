
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🤖 NLP Playground", layout="centered")
st.title("🤖 Text Classification Social media post")

# Task selection and user input
task = st.selectbox(
    "Choose a task",
    ["Sentiment Analysis", "Named Entity Recognition", "spam/ham"]
)

user_input = st.text_area("Enter your text here 👇", height=150)

if st.button("Run"):
    if not user_input:
        st.warning("Please enter some text!")
    else:
        if task == "Sentiment Analysis":
            analyzer = pipeline("sentiment-analysis")
            result = analyzer(user_input)[0]
            st.success(f"**{result['label']}** with confidence **{result['score']:.2%}**")

        elif task == "Named Entity Recognition":
            ner_tagger = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
            results = ner_tagger(user_input)
    
            if results:
              st.write("**Entities:**")
              for r in results:
                word = r['word']
                entity = r['entity_group']
                score = round(r['score'] * 100, 2)
                st.write(f"`{word}` → **{entity}** ({score}%)")
            else:
              st.warning("No entities found.")

        

        elif task == "spam/ham":
            classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
            label_map = {
                "LABEL_0": "ham",
                "LABEL_1": "spam"
            }
            result = classifier(user_input)
            for r in result:
                raw_label = r['label']
                label = label_map.get(raw_label, raw_label)
                score = round(r['score'] * 100, 2)
                st.info(f"**Prediction:** {label.capitalize()} ({score}%)")
