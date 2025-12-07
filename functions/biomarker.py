import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import time
import json

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# for predicting the risk of dementia in a patient, it seemed better just using a simple cnn rather than an lstm.
# we do not have access to an overall decreasing time series of a patients' health, and only indidividual health of each patient. 
# therefore, time sequences will actually hurt the model more than fix it.

# what we can do is clean the data, take out each other column that is not dementia [diabetes, hypercholesterolemia, age, etc]
# and add flags to them. a patient will be more at risk of dementia the more underlying conditions they have.
# it is much easier to do that way.

# sadly, no lstm implementation this time.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option("future.no_silent_downcasting", True)

def normalize(x):
    if pd.isna(x): # dne
        return 0
    x = str(x).strip().lower()
    if x in ["yes", "y", "true", "1", "present", "smoker"]:
        return 1
    else: return 0

def cleaning(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # conv col values to numeric
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
        df["gender"] = df["gender"].map({"male": 0, "female": 1})
        df["gender"] = df["gender"].fillna(0)

    if "smoking" in df.columns:
        df["smoking"] = df["smoking"].astype(str).str.strip().str.title()
        df["smoking"] = df["smoking"].replace({"None": 0, "Quit": 1, "Smoker": 2,}).fillna(0)

    # guarantees typos are inclusive
    yes_no_map = {
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        "YES": 1, "NO": 0,

        "Present": 1, "Absent": 0,
        "present": 1, "absent": 0,

        "Y": 1, "N": 0
    }

    # drop fazekas_cat col
    df = df.replace(yes_no_map)
    df = df.infer_objects(copy=False)
    df = df.drop(columns=["study_Name", "Fazekas_cat"], errors="ignore")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop dementia col
    df = df.fillna(0)
    y = df["dementia_all"].astype(np.float32).values
    X = df.drop(columns=["dementia_all", "dementia"], errors="ignore")  # drop both labels

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, list(X.columns)


class PatientDataset(Dataset):
    def __init__(self, x, y): # init tensors
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class DementiaNN(nn.Module): # simple predictor cnn model (3 fc layers, relu, sigmoid)

    def __init__(self, input_size, hidden_size_2=32, hidden_size_1=16, output_layer=1):
        super(DementiaNN, self).__init__()

        # nn.Sequential self init easily
        self.net = nn.Sequential( # multilayer cnn
            nn.Linear(input_size, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(hidden_size_2, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(hidden_size_1, output_layer),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
def train_model(model, train_loader, num_epochs=25, lr=0.001, pos_weight=None):
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"\nTraining time: {(end_time-start_time)/60:.2f} minutes")

def evaluate_model(model, test_loader): # evaluate on test set
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze()
            preds = (outputs >= 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    class_names = ["No Dementia", "Dementia"]
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

def risk_category(prob): # predictions of risk category based on probability
    if prob < 0.33:
        return "Low Risk"
    elif prob < 0.66:
        return "Moderate Risk"
    else:
        return "High Risk"
    
def predictor(model, scaler, column_order, patient_dict):
    df_new = pd.DataFrame([patient_dict])

    df_new["gender"] = df_new["gender"].map({"male": 0, "female": 1})
    df_new["smoking"] = df_new["smoking"].map({"None": 0, "Quit": 1, "Smoker": 2})

    for col in ["diabetes", "hypertension", "hypercholesterolemia", "Lacunes_Presence", "CMB_Presence", "dementia_all"]:
        if col in df_new.columns:
            df_new[col] = df_new[col].apply(normalize)

    if "fazekas_cat" not in df_new.columns:
        df_new["fazekas_cat"] = 0

    df_new = df_new[column_order]
    df_new_scaled = scaler.transform(df_new)

    x = torch.tensor(df_new_scaled, dtype=torch.float32)
    x = x.to(next(model.parameters()).device)
    
    with torch.no_grad():
        prob = model(x).item()

    risk_label = risk_category(prob)


    #debug
    # print("Columns expected:", column_order)
    # print("Columns in new patient:", df_new.columns.tolist())

    return prob, risk_label

def main(iterations, runs):
        records = "./Data/BMData/patient_records.csv"

        print("\nLoading data: ")
        X, y, scaler, col_order = cleaning(records)
        y = np.array(y, dtype=int)

        # train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_ds = PatientDataset(x_train, y_train)
        test_ds = PatientDataset(x_test, y_test)

        # BCEWithLogitsLoss -
        class_counts = np.bincount(y_train.astype(int))
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)

        # Create DataLoaders ---
        train_load = DataLoader(train_ds, batch_size=8, shuffle=True)
        test_load = DataLoader(test_ds, batch_size=8, shuffle=False)

        model = DementiaNN(input_size=x_train.shape[1])
        
        print("\nTraining model... \n")
        train_model(model, train_load, num_epochs=25, lr=0.0005, pos_weight=pos_weight)
        evaluate_model(model, test_load)

        with open("./research/example_patients.json", "r") as f:
            patients = json.load(f)

        # run multiple iterations
        with open("./research/biomarker_example_label.txt", "a") as f:
            for iteration in range(1, iterations + 1):
                f.write(f"------ [ ITERATION {iteration} ] ------\n\n")
                print(f"------ [ ITERATION {iteration} ] ------\n")
                
                for run in range(1, runs + 1):
                    f.write(f"Run {run}:\n")
                    print(f"Run {run}:")
                    
                    for patient in patients:
                        model.train()
                        risk_prob, risk_label = predictor(model, scaler, col_order, patient)
                        result = f"{patient.get('name', 'Unknown')} -> Predicted Dementia Risk: {risk_prob:.3f} ({risk_label})"
                        print(result)
                        f.write(result + "\n")
                    
                    f.write("\n")
                    print()
                f.write("\n")


if __name__ == "__main__":
    main(iterations=10, runs=3) # run 10 iterations for averaging purposes, 3 runs per