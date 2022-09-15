import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, EncoderDecoderModel

model_name = "IlyaGusev/rugpt3medium_sum_gazeta"
tokenizer_summary = AutoTokenizer.from_pretrained(model_name)
model_summary = AutoModelForCausalLM.from_pretrained(model_name)

# model_name_2 = "IlyaGusev/rubert_telegram_headlines"
# tokenizer_headline = AutoTokenizer.from_pretrained(model_name_2, do_lower_case=False, do_basic_tokenize=False, strip_accents=False)
# model_headline = EncoderDecoderModel.from_pretrained(model_name_2)

def summarization(summary_text: str):
    "Функция возвращает summary текста"
    text_tokens = tokenizer_summary( # токенизация текста
        summary_text,
        max_length=100,
        add_special_tokens=False,
        padding=False,
        truncation=True
    )['input_ids']

    input_ids = text_tokens + [tokenizer_summary.sep_token_id]
    input_ids = torch.LongTensor([input_ids])
    
    output_ids = model_summary.generate( #генерация текста
    input_ids=input_ids,
    max_length=100,
    num_beams=2,
    do_sample=True,
    temperature=10.,
    top_k=10,
    top_p=0.6,
    no_repeat_ngram_size=2,
    num_return_sequences=1)
    
    summary = tokenizer_summary.decode(output_ids[0], skip_special_tokens=False)
    summary = summary.split(tokenizer_summary.sep_token)[1]
    summary = summary.split(tokenizer_summary.eos_token)[0]
    
    end_punctuation = max(summary.rfind('.'), summary.rfind('?'),  summary.rfind('!'))
    ans = summary[:end_punctuation] + summary[end_punctuation]
    
    return ans

def headlining(headline_text: str):
    "Функция возвращает заголовок текста"
    input_ids = tokenizer_headline( # токенизация текста
        [article_text],
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    
    output_ids = model_headline.generate( #генерация заголовка
        input_ids=input_ids,
        max_length=64,
        no_repeat_ngram_size=3,
        num_beams=10,
        top_p=0.95
    )[0]
    
    headline = tokenizer_headline.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return headline
    
    
user_input = st.text_input("Скопируй текст, для которого ты хочешь сделать summary")

st.text_area("Скопируй текст, для которого ты хочешь сделать summary")
if user_input:
    st.write('Говно твой текст, другого summary у меня для тебя нет!')
    st.write(summarization(user_input))

