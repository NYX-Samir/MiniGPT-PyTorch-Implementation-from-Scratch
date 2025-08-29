
## MiniGPT — PyTorch from Scratch

Live Demo: https://minigpt-pytorch-implementation-from-scratch-nyx-samir.streamlit.app/

Minimal GPT-style decoder built in PyTorch with a Streamlit UI.

---

## Quick Start

git clone https://github.com/NYX-Samir/MiniGPT-PyTorch-Implementation-from-Scratch.git
cd MiniGPT-PyTorch-Implementation-from-Scratch
pip install -r requirements.txt

# Run Streamlit UI
streamlit run app.py

# Or CLI
python app.py --prompt "Hello world," --max_new_tokens 80


## Features

* Pure PyTorch GPT-style model (embeddings, masked self-attention, FFN, LayerNorm, residuals)
* Streamlit front-end for interactive generation
* Notebook (`mini-gpt.ipynb`) for exploration
* Pretrained weights (`model.pth`) for quick inference

---

## Structure

```
app.py        # Streamlit/CLI inference
model.py      # GPT model definition
mini-gpt.ipynb
model.pth
requirements.txt
```

---

## Example

```bash
python app.py --prompt "AI in 2026 will" --max_new_tokens 60 --temperature 0.8
```

Or open the [Live Demo](https://minigpt-pytorch-implementation-from-scratch-nyx-samir.streamlit.app/).

---

## License

MIT

```
```
