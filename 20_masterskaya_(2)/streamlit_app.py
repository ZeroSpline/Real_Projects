import pickle
import streamlit as st
import pandas as pd
import pysrt
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

stop_words = stopwords.words('english')
stop_words.remove('not')
lemmatizer = WordNetLemmatizer()

HTML = "<.*?>"
TAG = "{.*?}"
LETTERS = "[^a-zA-Z\'.,!? ]"
VECTORIZER = pickle.load(open('vect_model.pkl', 'rb'))
LOG_REG = pickle.load(open('log_reg.pkl', 'rb'))

def subs_clean_and_tokenize(subs):
    revi = re.sub(re.compile('<.*?>'), '', subs) # убираем html тэги
    revi = re.sub('[^A-Za-z0-9]+', ' ', revi) # оставляем слова
    revi = revi.lower() # изменяем регистр на строчной
    toke = nltk.word_tokenize(revi) # токенизация (разбиение текста на отдельные слова)
    revi = [word for word in toke if word not in stop_words] # убираем стоп слова
    revi = [lemmatizer.lemmatize(word) for word in revi] # лемматизация (начальные формы слов)
    revi = ' '.join(revi).split() # разделение слов
    revi = ' '.join(map(str, revi))
    return revi

def subs_vectorize(splitted_text):
    tfidf = pd.Series({0: splitted_text}) # преобразование в Series
    tfidf = VECTORIZER.transform(tfidf) # векторизация текста
    tfidf = pd.DataFrame.sparse.from_spmatrix(tfidf) # преобразование в dataframe
    return tfidf

def subs_stats(text):
    stats_list_solo = []
    words = str(text)
    # расчет количества слов
    count_words = len(words)
    stats_list_solo.append(count_words)
    # расчет количества уникальных слов
    unique_words = set(words)
    stats_list_solo.append(len(unique_words))
    # количество символов самого длинного слова
    sentense = words 
    w_dict = dict()
    for w in sentense.split(" "):
        w_dict[len(w)] = w
    biggest_word = w_dict[max(w_dict)]
    stats_list_solo.append(len(biggest_word))
    # количество слов с символами от 1 до 5
    symbols_1_5 = len(re.findall(r'\b(\w{1,5})\b', str(words)))
    stats_list_solo.append(symbols_1_5)  
    # количество слов с символами от 6 до 10    
    symbols_6_10 = len(re.findall(r'\b(\w{6,10})\b', str(words)))
    stats_list_solo.append(symbols_6_10)
    # количество слов с символами от 11 до 20    
    symbols_11_20 = len(re.findall(r'\b(\w{11,20})\b', str(words)))
    stats_list_solo.append(symbols_11_20)
    # количество слов начинающихся на гласные [aeiou] 
    symbols = len(re.findall(r'(\b[aeiou])', str(words)))
    stats_list_solo.append(symbols)
    stats_df = pd.DataFrame([stats_list_solo])
    stats_df.columns = ['stats_0', 'stats_1', 'stats_2', 'stats_3', 'stats_4', 'stats_5', 'stats_6']
    return stats_df
                        
def subs_union(df_text, df_stats):
    df_union = df_text.join(df_stats)
    return df_union                       

st.title('Оценка уровня английского языка киноленты по файлу с субтитрами')
st.text('Пожалуйста, загрузите английские субтитры к вашей киноленте.\n'
        'Затем нажмите кнопку, и алгоритмы машинного обучения расскажут вам,\n'
        'какой уровень английского необходим, чтобы понять киноленту.')
st.markdown('Вы можете скачать субтитры с https://subscene.com/.')
uploaded_file = st.file_uploader("**Загрузите файл субтитров в *\*.srt* формате**", type='srt')
if uploaded_file:
    f = uploaded_file.read()
    try:
        subtitles = pysrt.from_string(f.decode('utf-8'))
    except UnicodeDecodeError:
        try:
            subtitles = pysrt.from_string(f.decode('utf-16'))
        except UnicodeDecodeError:
            try:
                subtitles = pysrt.from_string(f.decode('latin-1'))
            except UnicodeDecodeError:
                st.text('Пожалуйста, проверьте кодировку. Обработка возможна для следующих кодировок: utf-16, utf-8 или latin-1')


predict_button = st.button('Оценить уровень английского языка')
if predict_button:
    try:
        X = subs_clean_and_tokenize(subtitles.text)
        Y = subs_vectorize(X)
        X = subs_stats(X)
        Z = subs_union(Y, X)

        predictions = pd.DataFrame({'Вероятность': LOG_REG.predict_proba(Z)[0],
                                    'Уровень английского языка': ['A2', 'B1', 'B2', 'C1']})

        msg = 'Вероятно, что ' + predictions.loc[predictions['Вероятность'].idxmax(), 'Уровень английского языка'] + \
              '- уровень подойдет для просмотра этой киноленты!'
        st.header(msg)
        st.text('Это диаграмма с вероятностями каждого уровня английского для вашей киноленты.')
        st.text('Уровни A1 и C2 пока не поддерживаются.')
        fig = plt.figure(figsize=(10, 4))
        sns.barplot(data=predictions, y='Вероятность', x='Уровень английского языка')
        st.pyplot(fig)
        
        st.text('Это статистики, собранные из файла с субтитрами для вашей киноленты.') 
        stats_data = pd.DataFrame({'Значения': X.loc[0],
                                   'Статистики': ['Количество слов', 'Количество уникальных слов', 
                                                  'Количество символов самого длинного слова', 
                                                  'Количество слов с символами от 1 до 5', 'Количество слов с символами от 6 до 10', 
                                                  'Количество слов с символами от 11 до 20', 
                                                  'Количество слов начинающихся на гласные A,E,I,O,U']})
        stats_data = stats_data.set_index('Статистики')
        st.write(stats_data)
        
    except NameError:
        st.text('Пожалуйста, сначала загрузите файл субтитров.')

       




