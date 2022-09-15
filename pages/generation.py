import streamlit as st
import os

best_len = 100
best_num_beams = 3
best_temp = 2

slider1 = st.slider('Длина предложения', 50, 200, best_len)
slider2 = st.slider('num_beams', 0, 7, best_num_beams)
slider3 = st.slider('Температура', 1, 10, best_temp)

user_input = st.text_input("Напиши начало строки, все остальное я сделаю сама")

import torch

def generation(user_input, len, beam, temp):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

    from transformers import GPT2LMHeadModel
    model_init = GPT2LMHeadModel.from_pretrained(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        output_attentions=False,
        output_hidden_states=False, )
    # weights_path = '/home/a_ladin/ds_offline/learning/project2/weights_of_preprocessing_text_Visotskii.pt'
    # model_init.load_state_dict(torch.load(weights_path))
    prompt = user_input
    prompt = tokenizer.encode(prompt, return_tensors='pt')
    out = model_init.generate(
        input_ids=prompt,
        max_length=len,
        num_beams=beam,
        do_sample=True,
        temperature=float(temp),
        top_k=10,
        top_p=0.6,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
    ).cpu().numpy()

    import textwrap
    # for out_ in out:
    #     return textwrap.fill(tokenizer.decode(out_), 100)
    #     # return textwrap.fill(tokenizer.decode(out_), 100), end='\n------------------\n'
    for out_ in out:
        text_dirty = textwrap.fill(tokenizer.decode(out_), 120)
    text_clear = text_dirty.split()
    for i, word in enumerate(text_clear):
        # the_end = i
        if ('.' in word) or ('!' in word) or ('?' in word):
            the_end = i + 1
    return (' '.join(text_clear[:the_end]))

if user_input:
    st.write(generation(user_input, slider1, slider2, slider3))