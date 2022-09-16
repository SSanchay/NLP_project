import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, EncoderDecoderModel

tokenizer = AutoTokenizer.from_pretrained("AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")
model = AutoModelForQuestionAnswering.from_pretrained("AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")


def QA(question: str, context: str):
    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()

    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]

    return tokenizer.decode(predict_answer_tokens)


if "my_input_question" not in st.session_state:
    st.session_state["my_input_question"] = ""

if "my_input_context" not in st.session_state:
    st.session_state["my_input_context"] = ""

st.title("Question answering")

my_input_question = st.text_input("", st.session_state["my_input_question"], key="question")
st.text("your question")

my_input_context = st.text_input("", st.session_state["my_input_context"], key="context")
st.text("context")
submit = st.button("Get answer")
if submit:
    st.session_state["my_input_question"] = my_input_question
    st.session_state["my_input_context"] = my_input_context
    st.subheader("Answer:")

    st.text_area(label="", value=QA(my_input_question, my_input_context), height=100)