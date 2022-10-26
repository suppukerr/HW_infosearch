import streamlit as st
from search_result import search
import time
import re
st.title("shure's search")


st.markdown(page_bg_img, unsafe_allow_html=True)
meas_names = ['bm25', 'bert']

page = st.radio('сначала выберите метрику', meas_names)

st.subheader('скорее вводи запрос')
query = st.text_input('давай')
if not query:
    st.warning('ты ничего не ввёл')
elif re.match(r'[A-Za-z]', query):
    st.warning('попробуй переформулировать на русском')
elif re.match(r'\d', query):
    st.warning('а есть вопросы, не связанные с числами. я не справляюсь')

if st.button('жми'):
    res, res_time = search(query, page)
    if not res:
        st.warning('по твоему запросу ничего не находит. где-то закралась ошибка')
    st.markdown(f'*поисковик искал {round(res_time, 3)} секунд*')
    for r in res:
        st.markdown(f'* {r}')
