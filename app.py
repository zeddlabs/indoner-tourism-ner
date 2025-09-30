import streamlit as st
from transformers import pipeline

# Load model & tokenizer yang sudah disimpan
@st.cache_resource
def load_model():
    ner = pipeline(
        "ner",
        model="ner_model",
        tokenizer="ner_model",
        aggregation_strategy=None
    )
    return ner

ner_pipeline = load_model()

st.title("📝 NER Bahasa Indonesia (IndoNER-Tourism)")
st.write("Website demo untuk Named Entity Recognition menggunakan IndoBERT + dataset IndoNER-Tourism.")

# Input teks
user_input = st.text_area("Masukkan kalimat bahasa Indonesia:", 
                          "Saya berkunjung ke Candi Borobudur di Jawa Tengah.")

if st.button("Proses NER"):
    if user_input.strip():
        results = ner_pipeline(user_input)

        st.subheader("Hasil Ekstraksi Entitas:")
        for r in results:
            st.markdown(f"- **{r['word']}** → `{r['entity']}` (score={r['score']:.2f})")
    else:
        st.warning("Masukkan kalimat terlebih dahulu.")
