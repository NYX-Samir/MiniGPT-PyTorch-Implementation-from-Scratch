# app.py
import streamlit as st
import torch
from model import MiniGPTModel, encode, decode, vocab_size, device, BLOCK_SIZE


@st.cache_resource
def load_model():
    model = MiniGPTModel().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

st.title("MiniGPT Text Generator")
st.write("A simple GPT-like model trained on Tiny Shakespeare.")


prompt = st.text_area("Enter your prompt:", "")

max_tokens = st.slider("Max new tokens to generate", 50, 500, 200)

if st.button("Generate"):
    try:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

        generated_idx = model.generate(context, max_new_tokens=max_tokens)
        generated_text = decode(generated_idx[0].tolist())

        st.subheader("Generated Text:")
        st.write(generated_text)

    except KeyError as e:
        st.error(f"Character not in vocabulary: {e}")
