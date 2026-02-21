import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import librosa
import os

# === FOLDERS CONTAINING WAV FILES ===
input_folders = [
    r"C:\Users\ahmed\Downloads\Pitt_Data\Dementia\wav\cookie",
    r"C:\Users\ahmed\Downloads\Pitt_Data\Control\wav\cookie"
]

# === OUTPUT FILE ===
output_file = os.path.join(os.path.expanduser("~"), r"Downloads\wav2vec_all_embeddings.pt")

# === LOAD Wav2Vec2 MODEL AND PROCESSOR ===
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# === DICTIONARY TO STORE ALL EMBEDDINGS ===
embeddings_dict = {}

# === LOOP THROUGH ALL WAV FILES IN BOTH FOLDERS ===
for folder in input_folders:
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            wav_path = os.path.join(folder, filename)
            
            # Load waveform
            waveform, sr = librosa.load(wav_path, sr=16000)
            waveform = torch.tensor(waveform)

            # Process and get embeddings
            inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)

            # Mean over sequence dimension
            audio_embedding = outputs.last_hidden_state.mean(dim=1)  # shape: (1, 768)

            # Use folder prefix to avoid duplicate filenames
            key_name = f"{os.path.basename(folder)}_{filename}"

            # Store in dictionary
            embeddings_dict[key_name] = audio_embedding.squeeze()  # remove batch dim

            print(f"✅ Processed: {key_name}")

# Save all embeddings as a single .pt file
torch.save(embeddings_dict, output_file)
print(f"\nAll embeddings saved as a single file:\n{output_file}")

# Print first 3 embeddings to see
for i, (key, emb) in enumerate(embeddings_dict.items()):
    print(f"\nFilename: {key}")
    print("Embedding shape:", emb.shape)
    print(emb[:10])  # print first 10 values of embedding for brevity
    if i >= 2:  # only show first 3 files
        break

