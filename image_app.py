import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO


def svd(img, top_k):
    U, sing_values, V = np.linalg.svd(img)
    sigma = np.zeros(shape=img.shape)
    np.fill_diagonal(sigma, sing_values)
    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    new_image = trunc_U @ trunc_sigma @ trunc_V
    return new_image


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
    image_raw = st.session_state.image

    image_show = np.array(image_raw)
    image = np.array(image_raw.convert('L'))

    st.caption(f'Разрешение изображения: {image_show.shape[0]}x{image_show.shape[1]}')
    st.image(image_show, caption='Оригинал', use_column_width=True)

    with st.sidebar.form(key='my_form'):
        top_k = st.slider('Выберите top K для сжатия', min_value=1, max_value=min(image.shape))
        choice = st.toggle('Цветное')
        submit = st.form_submit_button(label="Submit")

    if submit:
        st.caption(f'Вы выбрали top K: {top_k}')

        if top_k:
            with st.spinner(text="In progress..."):
                if not choice:
                    new_image = svd(image, top_k)

                else:
                    img_r = np.array(image_raw)[:, :, 0]
                    img_g = np.array(image_raw)[:, :, 1]
                    img_b = np.array(image_raw)[:, :, 2]

                    new_image_r = svd(img_r, top_k)
                    new_image_g = svd(img_g, top_k)
                    new_image_b = svd(img_b, top_k)

                    new_image = np.stack((new_image_r, new_image_g, new_image_b), axis=-1)

                new_image = (new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image))

            st.image(new_image, caption='Сжатое изображение', use_column_width=True)
