
import gradio as gr
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model and tokenizer
model_path = "distilbert_airline"  # your saved model folder
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Prediction function
def predict_intent(text):
    if not text.strip():
        return "Please enter some text."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=1).item()
    return model.config.id2label[predicted_id]

# Gradio interface
iface = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(lines=3, placeholder="Enter your query here..."),
    outputs="text",
    title="Airline Intent Classification"
)

# Launch app with public link
iface.launch(share=True)
