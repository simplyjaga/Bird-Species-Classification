import streamlit as st
import time
import prediction
import tensorflow as tf

st.sidebar.title("Bird Species Labeling")

num_species = st.radio("Number of Bird Species",('Two', 'Three'))
st.markdown(
    """
    <style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 58px;}
    </style>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file", label_visibility='hidden')
pred = st.sidebar.button('Run Model')

if uploaded_file is not None:
    st.audio(uploaded_file)

if pred:
    model = st.session_state['model']
    
    if num_species == 'Two':
        n = 2
    else:
        n = 3
        
    with st.spinner('Predicting labels...'):
        y_true, y_pred = prediction.demo(model, uploaded_file, n)
    st.success('Done!')
    labels = '_'.join(y_pred)
    st.write(labels)
    

if 'model' not in st.session_state:
    with st.spinner('Loading model...'):
        st.session_state['model'] = tf.keras.models.load_model('model')
    st.success('Done!')
    
