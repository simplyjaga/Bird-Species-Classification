import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import time
import prediction
import tensorflow as tf

name = {'lapwing' : 'Red Lapwing', 'sparrow' : 'House Sparrow',
        'Asiankoel': 'Asian Koel', 'goaway' : 'Grey Goaway', 'bluejay' : 'Blue Jay'}

st.set_page_config(layout = "wide", initial_sidebar_state = "expanded")
def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';
                          elements[i].style.} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

st.markdown("<h2 style='text-align:center;'><b>Bird Species Recognition Using Wav2Vec</b></h2>", unsafe_allow_html=True)
st.markdown('')

uploaded_file = st.file_uploader("Choose an audio sample", label_visibility='visible')
if uploaded_file is not None:
    st.audio(uploaded_file)
    
col1, col2 = st.columns(2)
with col1:
    st.markdown('')
    st.markdown('')
    pred = st.button('Run the Wav2Vec Model')
with col2:
    st.markdown('')
    num_species = st.radio("Number of bird species *",('Two', 'Three'))
    ChangeWidgetFontSize('Number of bird species *', '22px')

if pred:
    model = st.session_state['model']
    
    if num_species == 'Two':
        n = 2
    else:
        n = 3
        
    with st.spinner('Predicting...'):
        y_true, y_pred = prediction.demo(model, uploaded_file, n)
        
    st.success("Predicted bird species present in the audio sample")
    
    for i, c in enumerate(st.columns(n)):
        c.markdown(f"<h6>{name[y_pred[i]]}</h6>", unsafe_allow_html=True)
        image = Image.open(f"{y_pred[i]}.png")
        new_image = image.resize((100, 100))
        c.image(new_image, use_column_width=False)
    
if 'model' not in st.session_state:
    with st.spinner('Loading model...'):
        st.session_state['model'] = tf.keras.models.load_model('model')
    
