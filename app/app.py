
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuration
st.set_page_config(
    page_title="BrainScan",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 BrainScan")
st.write("Classification expérimentale d'images IRM cérébrales")

# Charger le modèle
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/brainscan_model.keras")

model = load_model()

# Ordre des classes utilisé pendant l'entraînement
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Importer une image
uploaded_file = st.file_uploader(
    "Importer une image IRM",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Image importée", width=300)

    if st.button("Analyser l'image"):
        # Préparer l'image
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)

        # Ajouter la dimension du batch
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction
        predictions = model.predict(image_array, verbose=0)
        predicted_index = np.argmax(predictions[0])

        predicted_class = classes[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        st.subheader("Résultat")
        st.write(f"Classe prédite : **{predicted_class}**")
        st.write(f"Score du modèle : **{confidence:.2f} %**")

        st.caption(
            "Démonstration académique uniquement. "
            "Ce modèle n'est pas un outil de diagnostic médical."
        )
