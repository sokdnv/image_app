import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.title('Сингулярный преобразователь')
st.caption('имени Сергея Кудинова')
st.divider()

uploaded_image = st.sidebar.file_uploader('Закинь свою картинку')
button = st.sidebar.button('Мне повезёт!')

if 'image' not in st.session_state:
    st.session_state.image = None

if button:
    response = requests.get('https://variety.com/wp-content/uploads/2021/07/Rick-Astley-Never-Gonna-Give-You-Up.png')
    url_image = Image.open(BytesIO(response.content))
    st.session_state.image = url_image

if uploaded_image:
    image = Image.open(uploaded_image)
    st.session_state.image = image

if st.session_state.image is not None:
    image = st.session_state.image

    image = np.array(image.convert('L'))

    st.caption(f'Разрешение изображения: {image.shape[0]}x{image.shape[1]}')
    st.image(image, caption='Оригинал', use_column_width=True)

    with st.sidebar.form(key='my_form'):
        top_k = st.slider('Выберите top K для сжатия', min_value=1, max_value=min(image.shape))
        submit = st.form_submit_button(label="Submit")

    if submit:
        st.caption(f'Вы выбрали top K: {top_k}')

    if top_k:
        with st.spinner(text="In progress..."):
            U, sing_values, V = np.linalg.svd(image)
            sigma = np.zeros(shape=image.shape)
            np.fill_diagonal(sigma, sing_values)

            trunc_U = U[:, :top_k]
            trunc_sigma = sigma[:top_k, :top_k]
            trunc_V = V[:top_k, :]

            new_image = trunc_U @ trunc_sigma @ trunc_V

            new_image = (new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image))

        st.image(new_image, caption='Сжатое изображение', use_column_width=True)
