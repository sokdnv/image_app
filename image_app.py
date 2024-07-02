import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO


def svd_compression(img, top_k):
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    compressed_img = (U[:, :top_k] @ np.diag(s[:top_k]) @ Vt[:top_k, :])
    return compressed_img


def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


st.title('Сингулярный преобразователь')
st.caption('имени Сергея Кудинова')
st.divider()

uploaded_image = st.sidebar.file_uploader('Закинь свою картинку')
button = st.sidebar.button('Мне повезёт!')

if 'image' not in st.session_state:
    st.session_state.image = None

if button:
    st.session_state.image = load_image('https://variety.com/wp-content/uploads/2021/07/Rick-Astley-Never-Gonna-Give-You-Up.png')

if uploaded_image:
    st.session_state.image = Image.open(uploaded_image)

if st.session_state.image is not None:
    image_raw = st.session_state.image
    image_array = np.array(image_raw)
    image_gray = np.array(image_raw.convert('L'))

    st.caption(f'Разрешение изображения: {image_array.shape[0]}x{image_array.shape[1]}')
    st.image(image_array, caption='Оригинал', use_column_width=True)

    with st.sidebar.form(key='compression_form'):
        top_k = st.slider('Выберите top K для сжатия', min_value=1, max_value=min(image_gray.shape))
        color_choice = st.toggle('Цветное')
        submit_button = st.form_submit_button(label="Сжать")

    if submit_button:
        st.caption(f'Вы выбрали top K: {top_k}')
        with st.spinner(text="Сжатие..."):
            if not color_choice:
                compressed_image = svd_compression(image_gray, top_k)
                compressed_image = (compressed_image - np.min(compressed_image)) / (np.max(compressed_image) - np.min(compressed_image))
                st.image(compressed_image, caption='Сжатое изображение (ЧБ)', use_column_width=True)
            else:
                channels = [image_array[:, :, i] for i in range(3)]
                compressed_channels = [svd_compression(channel, top_k) for channel in channels]
                compressed_image = np.stack(compressed_channels, axis=-1)
                compressed_image = (compressed_image - np.min(compressed_image)) / (np.max(compressed_image) - np.min(compressed_image))
                st.image(compressed_image, caption='Сжатое изображение (Цветное)', use_column_width=True)
