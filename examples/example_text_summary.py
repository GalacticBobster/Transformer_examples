from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

#device = torch.device("cuda" if torch.version.hip and torch.cuda.is_available() else "cpu")
#print("Using device:", device)

# Input text
text = """The Great Lakes are the largest group of freshwater lakes in the world by total area. 
They provide drinking water, transportation, and recreation for millions of people. 
However, they face challenges such as pollution, invasive species, and climate change impacts."""

# -------------------------------
# 1. Facebook BART
# -------------------------------
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
bart_summary = bart_summarizer(text, max_length=50, min_length=20, do_sample=False)[0]["summary_text"]

# -------------------------------
# 2. Google Pegasus
# -------------------------------
pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum")
pegasus_summary = pegasus_summarizer(text, max_length=50, min_length=20, do_sample=False)[0]["summary_text"]

# -------------------------------
# 3. GPT-2 (hacky summarization via prompt)
# -------------------------------
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = gpt2_tokenizer.encode("Summarize: " + text, return_tensors="pt")
gpt2_outputs = gpt2_model.generate(
    inputs,
    max_length=80,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)
gpt2_summary = gpt2_tokenizer.decode(gpt2_outputs[0], skip_special_tokens=True)

# -------------------------------
# Print results
# -------------------------------
print("=== BART Summary ===")
print(bart_summary)
print("\n=== Pegasus Summary ===")
print(pegasus_summary)
print("\n=== GPT-2 (hack) Summary ===")
print(gpt2_summary)

