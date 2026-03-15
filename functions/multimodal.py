import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms

from mri import MRI_CNN
from biomarker import Biomarker_NN
# from functions.speech import Speech_NN  # example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MRI model
mri_model = MRI_CNN()
mri_model.to(device).eval()

# Load Biomarker model
bio_model = Biomarker_NN(input_size=16)  # match your CSV features
bio_model.to(device).eval()


"""
Multimodal function that takes in all models, combines their predictions, and curates a final risk assessment.
"""

# --------------------------
# Multimodal Predictor
# --------------------------
class ModularMultimodalPredictor:
    def __init__(self, mri_model=None, bio_model=None, speech_model=None,weights={'mri':0.6, 'bio':0.4}):
        self.mri_model = mri_model
        self.bio_model = bio_model
        self.weights = weights

    def predict(self, mri_input=None, bio_input=None, speech_input=None):
        probs = []

        if self.mri_model and mri_input is not None:
            mri_input = mri_input.to(device)
            with torch.no_grad():
                p_mri = self.mri_model(mri_input).item()
            probs.append(self.weights['mri'] * p_mri)
        else:
            probs.append(0)

        if self.bio_model and bio_input is not None:
            bio_input = bio_input.to(device)
            with torch.no_grad():
                p_bio = self.bio_model(bio_input).item()
            probs.append(self.weights['bio'] * p_bio)
        else:
            probs.append(0)

        # sum weighted probabilities
        final_prob = sum(probs)
        final_prob = min(max(final_prob, 0.0), 1.0)  # clamp to [0,1]

        # simple risk categorization
        if final_prob < 0.33:
            risk_label = "Low Risk"
        elif final_prob < 0.66:
            risk_label = "Moderate Risk"
        else:
            risk_label = "High Risk"

        return final_prob, risk_label

def main(biomarker_csv_path=None, mri_path=None):
    predictor = ModularMultimodalPredictor(mri_model=mri_model, bio_model=bio_model)

    # biomarker in
    if biomarker_csv_path is None:
        biomarker_csv_path = input("Enter biomarker CSV path: ")

    df = pd.read_csv(biomarker_csv_path)
    bio_values = torch.tensor(df.values, dtype=torch.float32)

    # mri in
    if mri_path is None:
        mri_path = input("Enter MRI slice path: ")

    img = Image.open(mri_path).convert("L")  # grayscale
    transform = T.Compose([T.Resize((128,128)), T.ToTensor()])
    mri_input = torch.tensor(transform(img))  # add batch dimension
    
    prob, risk = predictor.predict(mri_input=mri_path, bio_input=biomarker_csv_path)
    print(f"Final Probability: {prob:.4f}, Risk Category: {risk}")

    risk_prob, risk_label = predictor.predict(mri_input=mri_input, bio_input=bio_values)

    return risk_prob, risk_label

if __name__ == "__main__":
    main("research/testing/MODERATE/moderate.json", "research/testing/MODERATE/OAS1_0308_MR1_mpr-1_110.jpg")