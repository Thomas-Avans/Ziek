import streamlit as st
import joblib

# === Laad model en vectorizer ===
model = joblib.load("model_ct_classifier.joblib")
vectorizer = joblib.load("vectorizer_ct_classifier.joblib")

# === Titel en uitleg ===
st.title("CT-indicatievoorspeller op basis van klinische kernwoorden")

st.markdown("""
Typ de klinische kernwoorden uit het verwijzingsveld, bijvoorbeeld:

`dyspneu`, `koorts`, `covid`, `verdenking longembolie`, `pijn`

De tool voorspelt de kans dat een **CT Angio Thorax** geïndiceerd is.
""")

# === Invoerveld ===
input_text = st.text_input("Voer kernwoorden in (spaties of komma's gescheiden):")

if input_text:
    # Vectoriseer de input
    X_input = vectorizer.transform([input_text])
    
    # Voorspel kans op CT
    ct_probability = model.predict_proba(X_input)[0][1] * 100

    # Toon resultaat
    st.metric(label="Waarschijnlijke noodzaak voor CT Angio Thorax", value=f"{ct_probability:.2f}%")
