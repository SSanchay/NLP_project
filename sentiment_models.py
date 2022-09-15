# для моделей
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from joblib import load
import re
import string
import numpy as np

import torch
import torch.nn as nn
import numpy as np
from json import load as json_load

import transformers


# Возможно нужны будут
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

## -------------------------- Для 1-ой модели ---------------------------

# Кэшируем стоп-слова
sw = stopwords.words('english')
#Загружаем vectorizer
tfid = load('tfidf.joblib')
#Загружаем модель для sentiment_ML
model_ml = load('logreg.joblib')

def clean(text):
    "Функция чистит текст"
    text = text.lower() #нижний регистр
    text = re.sub('<.*?>', '', text) # Remove HTML from text
    text = text.translate(str.maketrans('', '', string.punctuation)) # удаляем знаки препинания
    text = re.sub(r'\d+', ' ', text) # удаляем числа
    
    return text

def tokenize_review(review: str):
    "Функция чистит текст и возвращает токенизированные слова"
    cleaned_review = clean(review) # cleaning text
    wn_lemmatizer = WordNetLemmatizer() #lemmatization
    reg_tokenizer = RegexpTokenizer('\w+') #tokenization
    
    lemmatized_review = ' '.join([wn_lemmatizer.lemmatize(word, tag[0].lower()) # лемматизация с учётом части речи
                                if tag[0].lower() in ['a','n','v'] # word - проверка на adv, noun и verb
                                else wn_lemmatizer.lemmatize(word) # простая лемматизация
                                for word, tag in nltk.pos_tag(cleaned_review.split())]) # pos_tag - определяет часть речи
    
    tokenized_review = reg_tokenizer.tokenize_sents([lemmatized_review]) # ревью состоит из токенов
    clean_tokenized_review = ' '.join([word for word in tokenized_review[0] if word not in sw]) # удаляем стоп-слова, сверху объявили sw 
    
    return clean_tokenized_review

## -------------------------- Для 2-ой модели ---------------------------

class sentimentGRU(nn.Module):
    def __init__(self, 
                 vocab_size, # объём словаря слов
                 output_size, # нейроны полносвязного слоя
                 embedding_dim, # размер выходного эмбеддинга
                 hidden_dim, # размерность внутреннего слоя LSTM
                 n_layers, # число слоев в LSTM
                 drop_prob=0.5):
        super().__init__()

        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim,
                           hidden_dim,
                           n_layers,
                           dropout = drop_prob,
                           batch_first = True)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden):
        
        batch_size = x.size(0)
        embeds = self.embedding(x)
        
        gru_out, hidden = self.gru(embeds, hidden)
        gru_out = gru_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(gru_out)
        out = self.fc(out)
        
        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        
        return sig_out, hidden

    def init_hidden(self, batch_size):
        "Hidden state инициализируем нулями"
        hidden = torch.zeros((self.n_layers), batch_size, self.hidden_dim)
        
        return hidden
    
#загружаем словарь слово-число
with open('dict.json', 'r') as fp:
    vocab_to_int = json_load(fp)

#загружаем модель для sentiment_RNN
vocab_size = len(vocab_to_int) + 1 # размер словаря vocab_to_int
output_size = 1 # задача бинарной классификации
embedding_dim = 32 # размер слова
hidden_dim = 16 # размер вектора истории
n_layers = 2 # количество GRU слоев

model_rnn = sentimentGRU(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model_rnn.load_state_dict(torch.load('params_gru.pt', map_location=torch.device('cpu')))
model_rnn.eval()

## -------------------------- Для 3-ой модели ---------------------------

    
# DistilBERT: ## задаем саму модель
model_class = transformers.DistilBertModel
## токенайзер к ней (для некоторых моделей токенайзер будет отличаться, см. в документации к каждой модели конкретно)
tokenizer_class = transformers.DistilBertTokenizer

## загружаем веса для моделей
pretrained_weights = 'distilbert-base-uncased'

tokenizer_bert = tokenizer_class.from_pretrained(pretrained_weights)
model_bert = model_class.from_pretrained(pretrained_weights)
    
model_bert_ml = load('log_reg_2.joblib')

# 1-ая модель
def sentiment_ML(review: str):
    "функция обрабатывает текст и предсказывает на ML-модели"
    
    clean_tokenized_review = tokenize_review(review)
    tfid_representation = tfid.transform([clean_tokenized_review]) # ревью состоит из векторов, сверху загружаем vectorizer

    pred_proba = model_ml.predict_proba(tfid_representation) # хранит список предсказаний для двух классов 

    return np.round(pred_proba[0][1], 3) # берем первое и единственное предсказание и вероятность 1 класса (позитивный)

# 2-ая модель
def sentiment_RNN(review: str):
    "функция обрабатывает текст и предсказывает на RNN-модели"
    
    clean_tokenized_review = tokenize_review(review)
    
    num_review = [] # здесь будут вместо слов числа из словаря vocab_to_int #векторизуем
    for word in clean_tokenized_review.split(): 
        try:
            num_review.append(vocab_to_int[word]) 
        except KeyError as e:
            print(f'Word {word} not in dictionary!')
            
    #padding 
    padding_review = num_review[-200:] #обрубаем если больше 200 слов
    if len(num_review) <= 200:
        padding_review = list(np.zeros(200 - len(num_review))) + num_review #дополняем нулями если меньше 200 слов
        
    tensor_review = torch.Tensor(padding_review).long().unsqueeze(0) #создаем тензор ревью для модели
    test_h = model_rnn.init_hidden(1) #создаем hidden_state

    pred = model_rnn(tensor_review, test_h) 
    pred_proba = pred[0].item()
    
    return np.round(pred_proba, 3) # выдает вероятность 1 класса (позитивный)

# 3-ая модель
def sentiment_BERT_ML(review: str):
    "функция обрабатывает текст с помощью BERT и предсказывает на ML-модели"
    max_len = 64 # максимальная длина последовательности
    
    # применяем токенизатор
    tokenized = tokenizer_bert.encode(review,
                add_special_tokens=True, truncation=True, max_length=max_len) # добавили служебные токены и обрезали по макс длине
    
    padded = np.array(tokenized + [0]*(max_len-len(tokenized))) #дополняем нулями если меньше 64 слов
    
    attention_mask = np.where(padded != 0, 1, 0) # маскирование
    
    # переводим в тензоры
    input_ids = torch.tensor(padded).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    
    with torch.inference_mode():
        last_hidden_states = model_bert(input_ids, attention_mask=attention_mask)
    
    features = last_hidden_states[0][:, 0, :].numpy()
    
    pred_proba = model_bert_ml.predict_proba(features)
    
    
    return np.round(pred_proba[0][1], 3) # берем первое и единственное предсказание и вероятность 1 класса (позитивный)