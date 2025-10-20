import os
import torch
from transformers import BertTokenizer, BertModel


input_folders = [
    r"C:\Users\ahmed\Downloads\Pitt_Data\Dementia\transcripts\cookie",
    r"C:\Users\ahmed\Downloads\Pitt_Data\Control\transcripts\cookie"
]
# === OUTPUT FILE ===
output_file = os.path.join(os.path.expanduser("~"), "Downloads", "bert_all_embeddings.pt")

# === LOAD BERT MODEL AND TOKENIZER ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# === DICTIONARY TO STORE ALL EMBEDDINGS ===
embeddings_dict = {}

# === LOOP THROUGH ALL FOLDERS ===
for folder in input_folders:
    if not os.path.exists(folder):
        print(f"⚠️ Folder does not exist: {folder}")
        continue

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            txt_path = os.path.join(folder, filename)
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read()

            # Tokenize transcript
            inputs = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            # Use pooled_output as transcript embedding (1, 768)
            text_embedding = outputs.pooler_output.squeeze()  # shape: (768,)

            # Store in dictionary
            embeddings_dict[filename] = text_embedding

            print(f"✅ Processed: {filename}")

# Save all embeddings as a single .pt file
torch.save(embeddings_dict, output_file)
print(f"\nAll BERT embeddings saved as a single file:\n{output_file}")

# === PRINT SAMPLE OF OUTPUT ===
sample_keys = list(embeddings_dict.keys())[:3]  # first 3 files
print("\nSample embeddings:")
for key in sample_keys:
    print(f"{key}: {embeddings_dict[key][:10]} ...")  # show first 10 values
