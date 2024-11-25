import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re


# Funciones de limpieza y tokenización
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_text(text, tokenizer, max_length=64):
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_attention_mask=True,
        return_tensors='pt'
    )


def classify_text(text, model, tokenizer, device):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 'Texto inválido'

    if len(text) > 500:
        return 'Texto demasiado largo'

    clean = clean_text(text)
    tokens = tokenize_text(clean, tokenizer)

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
    except Exception as e:
        return f'Error al realizar la predicción: {str(e)}'

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).cpu().item()

    class_mapping = {
        0: 'Hate Speech',
        1: 'Offensive Language',
        2: 'Neither'
    }

    return class_mapping.get(predicted_class, 'Unknown')


# Cargar el modelo y el tokenizador
modelo_path = 'hate_speech_model'
tokenizer = BertTokenizer.from_pretrained(modelo_path)
model = BertForSequenceClassification.from_pretrained(modelo_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Configurar la aplicación Streamlit
st.title("Detector de Hate Speech y Lenguaje Ofensivo")
st.write("Introduce un texto y el modelo lo clasificará.")

texto_usuario = st.text_area("Introduce el texto aquí:")

if st.button("Clasificar"):
    resultado = classify_text(texto_usuario, model, tokenizer, device)
    st.write(f"**Clasificación:** {resultado}")