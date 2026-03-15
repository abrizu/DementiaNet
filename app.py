from pathlib import Path
import sys
import json
import shutil
import datetime

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(layout="wide", page_title="DementiaNet", page_icon="🧠")

BASE_DIR   = Path(__file__).parent
CKPT_DIR   = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
REGISTRY   = CKPT_DIR / "registry.json"

sys.path.insert(0, str(BASE_DIR / "functions"))

SPE_CKPT = CKPT_DIR / "speech_best.pt"

_DEFAULT_MRI_CKPT = str(CKPT_DIR / "mri_best.pt")
_DEFAULT_BIO_CKPT = str(CKPT_DIR / "biomarker_best.pt")


# ── Registry helpers ──────────────────────────────────────────────────────────
def load_registry() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text())
        except Exception:
            pass
    return {"mri": [], "biomarker": []}


def save_registry(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=2))


def register_checkpoint(
    modality: str, name: str, filename: str,
    val_acc: float, epochs: int, lr: float, notes: str,
) -> None:
    reg = load_registry()
    reg.setdefault(modality, [])
    reg[modality] = [e for e in reg[modality] if e["name"] != name]
    reg[modality].insert(0, {
        "name":    name,
        "file":    filename,
        "val_acc": round(val_acc * 100, 1),
        "epochs":  epochs,
        "lr":      lr,
        "date":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes":   notes,
    })
    save_registry(reg)


def delete_checkpoint_entry(modality: str, name: str) -> None:
    reg     = load_registry()
    entries = reg.get(modality, [])
    target  = next((e for e in entries if e["name"] == name), None)
    if target:
        pt = CKPT_DIR / target["file"]
        if pt.exists() and target["file"] not in ("mri_best.pt", "biomarker_best.pt"):
            pt.unlink()
        reg[modality] = [e for e in entries if e["name"] != name]
        save_registry(reg)


# ── Session-state defaults ────────────────────────────────────────────────────
if "mri_active_ckpt" not in st.session_state:
    st.session_state["mri_active_ckpt"] = _DEFAULT_MRI_CKPT
if "bio_active_ckpt" not in st.session_state:
    st.session_state["bio_active_ckpt"] = _DEFAULT_BIO_CKPT


# ── Model architectures ───────────────────────────────────────────────────────
class _MRI_CNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.norm  = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.4)
        self.fc1   = nn.Linear(32 * 32 * 32, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class _Biomarker_NN(nn.Module):
    def __init__(self, input_size: int, hidden_size_2: int = 32,
                 hidden_size_1: int = 16, output_layer: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_2), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden_size_2, hidden_size_1), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden_size_1, output_layer), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Path-based cached loaders ─────────────────────────────────────────────────
@st.cache_resource
def load_mri_model(ckpt_path: str):
    path = Path(ckpt_path)
    if not path.exists():
        return None
    try:
        try:
            from mri import MRI_CNN
        except Exception:
            MRI_CNN = _MRI_CNN
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = MRI_CNN(num_classes=4)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:
        st.sidebar.warning(f"MRI model failed to load: {exc}")
        return None


@st.cache_resource
def load_biomarker_model(ckpt_path: str):
    path = Path(ckpt_path)
    if not path.exists():
        return None
    try:
        state      = torch.load(path, map_location="cpu", weights_only=True)
        input_size = state["net.0.weight"].shape[1]
        try:
            from biomarker import Biomarker_NN
        except Exception:
            Biomarker_NN = _Biomarker_NN
        model = Biomarker_NN(input_size=input_size)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:
        st.sidebar.warning(f"Biomarker model failed to load: {exc}")
        return None


@st.cache_resource
def load_speech_model():
    try:
        from speech import SpeechModel  # type: ignore
        if not SPE_CKPT.exists():
            return None
        state = torch.load(SPE_CKPT, map_location="cpu", weights_only=True)
        model = SpeechModel()
        model.load_state_dict(state)
        model.eval()
        return model
    except (ImportError, AttributeError):
        return None
    except Exception:
        return None


mri_model = load_mri_model(st.session_state["mri_active_ckpt"])
bio_model = load_biomarker_model(st.session_state["bio_active_ckpt"])
spe_model = load_speech_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DementiaNet")
    st.caption("Multimodal Dementia Risk Prediction")
    st.divider()

    st.subheader("Model Status")
    _reg = load_registry()

    def _active_label(modality: str, active_path: str) -> str:
        p       = Path(active_path)
        entries = _reg.get(modality, [])
        match   = next((e for e in entries if e["file"] == p.name), None)
        return match["name"] if match else p.stem

    mri_ready = Path(st.session_state["mri_active_ckpt"]).exists()
    bio_ready = Path(st.session_state["bio_active_ckpt"]).exists()

    st.markdown(f"**MRI Model** &nbsp;&nbsp; {'✅ Ready' if mri_ready else '❌ Not trained'}")
    if mri_ready:
        st.caption(f"Active: `{_active_label('mri', st.session_state['mri_active_ckpt'])}`")

    st.markdown(f"**Biomarker** &nbsp;&nbsp; {'✅ Ready' if bio_ready else '❌ Not trained'}")
    if bio_ready:
        st.caption(f"Active: `{_active_label('biomarker', st.session_state['bio_active_ckpt'])}`")

    st.markdown(f"**Speech** &nbsp;&nbsp; {'✅ Ready' if SPE_CKPT.exists() else '❌ Not available yet'}")

    st.divider()
    st.caption(
        "⚠️ Research use only. Not a validated clinical diagnostic instrument. "
        "Do not use for actual medical decision-making."
    )


# ── Shared training helper ────────────────────────────────────────────────────
def _epoch_log(bar, log_slot, epoch, num_epochs, avg_loss, acc, best_acc):
    pct     = (epoch + 1) / num_epochs
    is_best = acc > best_acc
    bar.progress(pct, text=f"Epoch {epoch+1}/{num_epochs}")
    log_slot.markdown(
        f"**Epoch {epoch+1}/{num_epochs}** — "
        f"loss `{avg_loss:.4f}` | val acc `{acc*100:.1f}%`"
        + (" ⭐ new best" if is_best else "")
    )
    st.write(
        f"Epoch {epoch+1:>2}/{num_epochs} | loss={avg_loss:.4f} | acc={acc*100:.1f}%"
        + (" ← best" if is_best else "")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MRI — training, save dialog, checkpoint manager, full panel
# ═══════════════════════════════════════════════════════════════════════════════
def _run_mri_training(data_dir: Path, num_epochs: int, lr: float) -> None:
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset  = datasets.ImageFolder(root=str(data_dir), transform=xform)
    n_cls    = len(dataset.classes)
    train_sz = int(0.8 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_sz, len(dataset) - train_sz],
        generator=torch.Generator().manual_seed(42),
    )

    targets     = [dataset.targets[i] for i in train_ds.indices]
    class_cnt   = np.bincount(targets).astype(float)
    class_w     = 1.0 / np.sqrt(class_cnt)
    class_w    /= class_w.sum()
    sampler     = WeightedRandomSampler([class_w[t] for t in targets], len(targets), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,   num_workers=0)

    model     = _MRI_CNN(num_classes=n_cls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    temp_path = CKPT_DIR / "mri_training_session.pt"

    bar      = st.progress(0, text="Starting training…")
    log_slot = st.empty()
    best_acc = 0.0

    with st.status("Training MRI model…", expanded=True) as status:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds    = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)
            acc = correct / total

            _epoch_log(bar, log_slot, epoch, num_epochs, avg_loss, acc, best_acc)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), temp_path)

        status.update(
            label=f"Training complete — best val accuracy: {best_acc*100:.1f}%",
            state="complete",
        )

    st.session_state["mri_pending_save"] = {
        "acc": best_acc, "epochs": num_epochs, "lr": lr,
        "temp_file": str(temp_path),
    }
    st.rerun()


def _mri_save_dialog() -> None:
    pending = st.session_state.get("mri_pending_save")
    if not pending:
        return

    st.success(f"✅ Training complete — best val accuracy: **{pending['acc']*100:.1f}%**")
    st.subheader("💾 Save Checkpoint")

    default_name = f"mri_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    col_a, col_b = st.columns([1, 2])
    with col_a:
        save_name  = st.text_input("Checkpoint name", value=default_name, key="mri_save_name")
    with col_b:
        save_notes = st.text_input(
            "Notes (optional)", key="mri_save_notes",
            placeholder=f"{pending['epochs']} epochs, lr={pending['lr']}",
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save & Activate", type="primary", key="mri_do_save"):
            slug     = (save_name.strip() or default_name).replace(" ", "_")
            filename = f"mri_{slug}.pt"
            dest     = CKPT_DIR / filename
            shutil.copy(pending["temp_file"], dest)
            shutil.copy(pending["temp_file"], CKPT_DIR / "mri_best.pt")
            register_checkpoint(
                "mri", slug, filename,
                pending["acc"], pending["epochs"], pending["lr"], save_notes.strip(),
            )
            st.session_state["mri_active_ckpt"] = str(dest)
            del st.session_state["mri_pending_save"]
            st.cache_resource.clear()
            st.rerun()
    with c2:
        if st.button("⏭ Skip (don't save name)", key="mri_skip_save"):
            shutil.copy(pending["temp_file"], CKPT_DIR / "mri_best.pt")
            st.session_state["mri_active_ckpt"] = _DEFAULT_MRI_CKPT
            del st.session_state["mri_pending_save"]
            st.cache_resource.clear()
            st.rerun()


def _mri_checkpoint_table() -> None:
    import pandas as pd
    reg     = load_registry()
    entries = reg.get("mri", [])

    if not entries:
        st.info("No saved checkpoints yet. Use the **Train New Model** tab to create one.")
        return

    active_file = Path(st.session_state["mri_active_ckpt"]).name
    rows = [
        {
            "Active": "✅" if e["file"] == active_file else "",
            "Name":    e["name"],
            "Val Acc": f"{e['val_acc']}%",
            "Epochs":  e["epochs"],
            "LR":      e["lr"],
            "Date":    e["date"],
            "Notes":   e.get("notes", ""),
        }
        for e in entries
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    names   = [e["name"] for e in entries]
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Load a checkpoint**")
        load_name = st.selectbox("Select", names, key="mri_load_sel", label_visibility="collapsed")
        if st.button("⬆️ Load & Activate", key="mri_load_btn", use_container_width=True):
            entry = next(e for e in entries if e["name"] == load_name)
            path  = CKPT_DIR / entry["file"]
            if path.exists():
                st.session_state["mri_active_ckpt"] = str(path)
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(f"`{entry['file']}` not found on disk.")

    with col_r:
        st.markdown("**Delete a checkpoint**")
        del_name = st.selectbox("Select", names, key="mri_del_sel", label_visibility="collapsed")
        if st.button("🗑 Delete", key="mri_del_btn", type="secondary", use_container_width=True):
            entry = next((e for e in entries if e["name"] == del_name), None)
            if entry and entry["file"] == active_file:
                st.warning("Can't delete the active checkpoint. Load a different one first.")
            else:
                delete_checkpoint_entry("mri", del_name)
                st.rerun()


def _mri_training_form() -> None:
    data_dir = BASE_DIR / "Data" / "MRIData" / "Data"
    if not data_dir.exists():
        st.error("Dataset folder not found at `Data/MRIData/Data/`.")
        st.markdown(
            """
**Expected layout:**
```
Data/MRIData/Data/
├── Mild Dementia/
├── Very mild Dementia/
├── Moderate Dementia/
└── Non Demented/
```
Download from [Kaggle – Alzheimer MRI (4-class)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images-updates).
            """
        )
        return

    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    counts  = {c: len(list((data_dir / c).glob("*.*"))) for c in classes}
    total   = sum(counts.values())

    st.success(f"Dataset ready — **{total:,}** images across **{len(classes)}** classes")
    for col, (cls, cnt) in zip(st.columns(len(classes)), counts.items()):
        col.metric(cls, cnt)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        num_epochs = st.slider("Epochs", 3, 50, 10, key="mri_epochs")
    with col_b:
        lr_choice = st.select_slider(
            "Learning Rate",
            options=[0.00005, 0.0001, 0.0005, 0.001],
            format_func=lambda x: f"{x:.5f}",
            value=0.0001,
            key="mri_lr",
        )

    if st.button("🚀 Start Training", key="mri_train_btn", type="primary", use_container_width=True):
        _run_mri_training(data_dir, num_epochs, lr_choice)


def _mri_manager_panel() -> None:
    if "mri_pending_save" in st.session_state:
        _mri_save_dialog()
        return

    active_exists = Path(st.session_state["mri_active_ckpt"]).exists()
    with st.expander(
        "⚙️ Model Manager" + ("  —  ⚠️ no model loaded" if not active_exists else ""),
        expanded=not active_exists,
    ):
        ckpt_tab, train_tab = st.tabs(["📁 Saved Checkpoints", "🏋️ Train New Model"])
        with ckpt_tab:
            _mri_checkpoint_table()
        with train_tab:
            _mri_training_form()


# ═══════════════════════════════════════════════════════════════════════════════
# Biomarker — training, save dialog, checkpoint manager, full panel
# ═══════════════════════════════════════════════════════════════════════════════
def _load_bio_csv() -> "Path | None":
    for candidate in [
        BASE_DIR / "Data" / "BMData" / "main.csv",
        BASE_DIR / "Data" / "BMData" / "patient_records.csv",
    ]:
        if candidate.exists():
            return candidate
    return None


def _clean_bio_df(csv_path: Path):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    yes_no = {
        "yes": 1, "no": 0, "Yes": 1, "No": 0, "YES": 1, "NO": 0,
        "Present": 1, "Absent": 0, "present": 1, "absent": 0, "Y": 1, "N": 0,
    }
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    pd.set_option("future.no_silent_downcasting", True)

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
        df["gender"] = df["gender"].map({"male": 0, "female": 1}).fillna(0)
    if "smoking" in df.columns:
        df["smoking"] = (
            df["smoking"].astype(str).str.strip().str.title()
            .replace({"None": 0, "Quit": 1, "Smoker": 2}).fillna(0)
        )

    df = df.replace(yes_no).infer_objects(copy=False)
    df = df.drop(columns=["study_Name", "Fazekas_cat"], errors="ignore")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    y = df["dementia_all"].astype(np.float32).values
    X = df.drop(columns=["dementia_all", "dementia"], errors="ignore")
    return MinMaxScaler().fit_transform(X), y


def _run_biomarker_training(csv_path: Path, num_epochs: int, lr: float) -> None:
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

    X_scaled, y = _clean_bio_df(csv_path)
    x_tr, x_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y.astype(int)
    )

    class _DS(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return self.x[i], self.y[i]

    train_loader = DataLoader(_DS(x_tr, y_tr), batch_size=8, shuffle=True)
    test_loader  = DataLoader(_DS(x_te, y_te), batch_size=8, shuffle=False)

    counts     = np.bincount(y_tr.astype(int))
    pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = _Biomarker_NN(input_size=x_tr.shape[1]).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    temp_path  = CKPT_DIR / "biomarker_training_session.pt"

    bar      = st.progress(0, text="Starting training…")
    log_slot = st.empty()
    best_acc = 0.0

    with st.status("Training Biomarker model…", expanded=True) as status:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb).squeeze(), yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds    = (torch.sigmoid(model(xb).squeeze()) >= 0.5).long()
                    correct += (preds == yb.long()).sum().item()
                    total   += yb.size(0)
            acc = correct / total

            _epoch_log(bar, log_slot, epoch, num_epochs, avg_loss, acc, best_acc)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), temp_path)

        status.update(
            label=f"Training complete — best val accuracy: {best_acc*100:.1f}%",
            state="complete",
        )

    st.session_state["bio_pending_save"] = {
        "acc": best_acc, "epochs": num_epochs, "lr": lr,
        "temp_file": str(temp_path),
    }
    st.rerun()


def _bio_save_dialog() -> None:
    pending = st.session_state.get("bio_pending_save")
    if not pending:
        return

    st.success(f"✅ Training complete — best val accuracy: **{pending['acc']*100:.1f}%**")
    st.subheader("💾 Save Checkpoint")

    default_name = f"bio_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    col_a, col_b = st.columns([1, 2])
    with col_a:
        save_name  = st.text_input("Checkpoint name", value=default_name, key="bio_save_name")
    with col_b:
        save_notes = st.text_input(
            "Notes (optional)", key="bio_save_notes",
            placeholder=f"{pending['epochs']} epochs, lr={pending['lr']}",
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save & Activate", type="primary", key="bio_do_save"):
            slug     = (save_name.strip() or default_name).replace(" ", "_")
            filename = f"bio_{slug}.pt"
            dest     = CKPT_DIR / filename
            shutil.copy(pending["temp_file"], dest)
            shutil.copy(pending["temp_file"], CKPT_DIR / "biomarker_best.pt")
            register_checkpoint(
                "biomarker", slug, filename,
                pending["acc"], pending["epochs"], pending["lr"], save_notes.strip(),
            )
            st.session_state["bio_active_ckpt"] = str(dest)
            del st.session_state["bio_pending_save"]
            st.cache_resource.clear()
            st.rerun()
    with c2:
        if st.button("⏭ Skip (don't save name)", key="bio_skip_save"):
            shutil.copy(pending["temp_file"], CKPT_DIR / "biomarker_best.pt")
            st.session_state["bio_active_ckpt"] = _DEFAULT_BIO_CKPT
            del st.session_state["bio_pending_save"]
            st.cache_resource.clear()
            st.rerun()


def _bio_checkpoint_table() -> None:
    import pandas as pd
    reg     = load_registry()
    entries = reg.get("biomarker", [])

    if not entries:
        st.info("No saved checkpoints yet. Use the **Train New Model** tab to create one.")
        return

    active_file = Path(st.session_state["bio_active_ckpt"]).name
    rows = [
        {
            "Active": "✅" if e["file"] == active_file else "",
            "Name":    e["name"],
            "Val Acc": f"{e['val_acc']}%",
            "Epochs":  e["epochs"],
            "LR":      e["lr"],
            "Date":    e["date"],
            "Notes":   e.get("notes", ""),
        }
        for e in entries
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    names    = [e["name"] for e in entries]
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Load a checkpoint**")
        load_name = st.selectbox("Select", names, key="bio_load_sel", label_visibility="collapsed")
        if st.button("⬆️ Load & Activate", key="bio_load_btn", use_container_width=True):
            entry = next(e for e in entries if e["name"] == load_name)
            path  = CKPT_DIR / entry["file"]
            if path.exists():
                st.session_state["bio_active_ckpt"] = str(path)
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(f"`{entry['file']}` not found on disk.")

    with col_r:
        st.markdown("**Delete a checkpoint**")
        del_name = st.selectbox("Select", names, key="bio_del_sel", label_visibility="collapsed")
        if st.button("🗑 Delete", key="bio_del_btn", type="secondary", use_container_width=True):
            entry = next((e for e in entries if e["name"] == del_name), None)
            if entry and entry["file"] == active_file:
                st.warning("Can't delete the active checkpoint. Load a different one first.")
            else:
                delete_checkpoint_entry("biomarker", del_name)
                st.rerun()


def _bio_training_form() -> None:
    import pandas as pd
    csv_path = _load_bio_csv()
    if csv_path is None:
        st.error("Dataset CSV not found at `Data/BMData/main.csv`.")
        st.markdown(
            "Download from [Kaggle – Comprehensive Health & Brain Imaging Dataset]"
            "(https://www.kaggle.com/datasets/snmahsa/comprehensive-health-and-brain-imaging-dataset) "
            "and place the CSV at `Data/BMData/main.csv`."
        )
        return

    df_info = pd.read_csv(csv_path)
    n_rows  = len(df_info)
    n_pos   = int((df_info["dementia_all"] != 0).sum()) if "dementia_all" in df_info.columns else "?"
    n_neg   = (n_rows - n_pos) if isinstance(n_pos, int) else "?"

    st.success(f"Dataset ready — **{n_rows:,}** patient records in `{csv_path.name}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", f"{n_rows:,}")
    c2.metric("Dementia +", f"{n_pos:,}" if isinstance(n_pos, int) else n_pos)
    c3.metric("Dementia −", f"{n_neg:,}" if isinstance(n_neg, int) else n_neg)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        num_epochs = st.slider("Epochs", 5, 100, 25, key="bio_epochs")
    with col_b:
        lr_choice = st.select_slider(
            "Learning Rate",
            options=[0.00005, 0.0001, 0.0005, 0.001, 0.005],
            format_func=lambda x: f"{x:.5f}",
            value=0.0005,
            key="bio_lr",
        )

    if st.button("🚀 Start Training", key="bio_train_btn", type="primary", use_container_width=True):
        _run_biomarker_training(csv_path, num_epochs, lr_choice)


def _bio_manager_panel() -> None:
    if "bio_pending_save" in st.session_state:
        _bio_save_dialog()
        return

    active_exists = Path(st.session_state["bio_active_ckpt"]).exists()
    with st.expander(
        "⚙️ Model Manager" + ("  —  ⚠️ no model loaded" if not active_exists else ""),
        expanded=not active_exists,
    ):
        ckpt_tab, train_tab = st.tabs(["📁 Saved Checkpoints", "🏋️ Train New Model"])
        with ckpt_tab:
            _bio_checkpoint_table()
        with train_tab:
            _bio_training_form()


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🧠 MRI", "🩺 Biomarkers", "🎙️ Speech", "📊 Ensemble"]
)


# ──────────────────────────────────────────────────────────────────────────────
# MRI Tab
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("MRI Analysis")
    _mri_manager_panel()
    st.divider()

    uploaded_mri = st.file_uploader(
        "Upload an MRI scan", type=["png", "jpg", "jpeg"], key="mri_upload"
    )

    if uploaded_mri is not None:
        col_img, col_res = st.columns([1, 2])

        with col_img:
            st.image(uploaded_mri, use_container_width=True)

        with col_res:
            analyze_disabled = mri_model is None
            if st.button(
                "Analyze MRI", key="mri_btn", disabled=analyze_disabled,
                help="Train or load a model using the Model Manager above." if analyze_disabled else None,
            ):
                import plotly.graph_objects as go
                import torchvision.transforms as T
                from PIL import Image

                img   = Image.open(uploaded_mri).convert("L")
                xform = T.Compose([
                    T.Resize((128, 128)),
                    T.ToTensor(),
                    T.Normalize((0.5,), (0.5,)),
                ])
                tensor = xform(img).unsqueeze(0)

                with torch.no_grad():
                    logits = mri_model(tensor)
                    probs  = torch.softmax(logits, dim=1).squeeze().tolist()

                CLASSES    = ["Non-Demented", "Very Mild", "Mild", "Moderate"]
                BAR_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
                pred_idx   = int(np.argmax(probs))
                pred_label = CLASSES[pred_idx]
                pred_prob  = probs[pred_idx]

                RISK_W     = [0.0, 0.33, 0.67, 1.0]
                risk_score = float(sum(p * w for p, w in zip(probs, RISK_W)))

                fig = go.Figure(go.Bar(
                    x=probs, y=CLASSES, orientation="h",
                    marker_color=BAR_COLORS,
                    text=[f"{p * 100:.1f}%" for p in probs],
                    textposition="auto",
                ))
                fig.update_layout(
                    title="Class Probabilities",
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    height=300, margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                msg = f"**Predicted: {pred_label}** ({pred_prob * 100:.1f}% confidence)"
                if pred_label == "Non-Demented":
                    st.success(msg)
                elif pred_label == "Very Mild":
                    st.warning(msg)
                else:
                    st.error(msg)

                st.session_state["mri_result"] = {
                    "label":     pred_label,
                    "probability": risk_score,
                    "all_probs": dict(zip(CLASSES, probs)),
                }

        saliency = CKPT_DIR / "mri_saliency.png"
        if saliency.exists():
            st.image(str(saliency), caption="Gradient Saliency — regions influencing prediction")


# ──────────────────────────────────────────────────────────────────────────────
# Biomarker Tab
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Biomarker Analysis")
    _bio_manager_panel()
    st.divider()

    with st.form("biomarker_form"):
        col_l, col_r = st.columns(2)

        with col_l:
            age                  = st.number_input("Age", min_value=0, max_value=120, value=65, step=1)
            gender               = st.selectbox("Gender", ["Male", "Female"])
            diabetes             = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension         = st.selectbox("Hypertension", ["No", "Yes"])
            hypercholesterolemia = st.selectbox("Hypercholesterolemia", ["No", "Yes"])

        with col_r:
            fazekas     = st.slider("Fazekas Score", min_value=0, max_value=3, value=0)
            lacunes     = st.number_input("Lacunes", min_value=0, max_value=20, value=0, step=1)
            microbleeds = st.number_input("Microbleeds", min_value=0, max_value=30, value=0, step=1)
            mmse        = st.slider("MMSE Score", min_value=0, max_value=30, value=28)

        submitted = st.form_submit_button("Analyze Biomarkers")

    if submitted:
        if bio_model is None:
            st.warning("Biomarker model not loaded. Use the Model Manager above to train or load one.")
        else:
            import plotly.graph_objects as go

            gender_enc = 1 if gender == "Male" else 0
            diab_enc   = 1 if diabetes == "Yes" else 0
            hyp_enc    = 1 if hypertension == "Yes" else 0
            hchol_enc  = 1 if hypercholesterolemia == "Yes" else 0

            features = np.array([[
                age / 120, gender_enc, diab_enc, hyp_enc, hchol_enc,
                fazekas / 3, lacunes / 20, microbleeds / 30, (30 - mmse) / 30,
            ]], dtype=np.float32)

            expected = bio_model.net[0].in_features
            if features.shape[1] < expected:
                features = np.pad(features, ((0, 0), (0, expected - features.shape[1])))
            else:
                features = features[:, :expected]

            x = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                prob = float(bio_model(x).item())

            risk_label = "Low Risk" if prob < 0.33 else ("Moderate Risk" if prob < 0.66 else "High Risk")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": f"Dementia Risk Score — {risk_label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#2c3e50"},
                    "steps": [
                        {"range": [0, 40],   "color": "#2ecc71"},
                        {"range": [40, 70],  "color": "#f1c40f"},
                        {"range": [70, 100], "color": "#e74c3c"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": prob * 100},
                },
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.session_state["biomarker_result"] = {"label": risk_label, "probability": prob}


# ──────────────────────────────────────────────────────────────────────────────
# Speech Tab
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Speech Analysis")

    audio_file = st.file_uploader(
        "Upload a speech recording", type=["wav", "mp3"], key="audio_upload"
    )
    transcript = st.text_area("Transcript (optional — paste if available)")

    if st.button("Analyze Speech", key="spe_btn"):
        if spe_model is None:
            st.info("Speech model is still being integrated. Check back after `functions/speech.py` is complete.")
        elif audio_file is None:
            st.warning("Please upload an audio file first.")
        else:
            try:
                import io
                import librosa

                audio_bytes = audio_file.read()
                y_audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

                st.subheader("Waveform")
                step = max(1, len(y_audio) // 2000)
                st.line_chart(y_audio[::step])

                with torch.no_grad():
                    raw_out  = spe_model(torch.zeros(1, 1, 128))
                    spe_prob = float(torch.sigmoid(raw_out).squeeze())

                spe_label = "High Risk" if spe_prob > 0.5 else "Low Risk"
                st.success(f"Speech analysis complete — **{spe_label}** ({spe_prob * 100:.1f}% confidence)")
                st.session_state["speech_result"] = {"label": spe_label, "probability": spe_prob}
            except Exception as exc:
                st.error(f"Speech analysis failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Tab
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Ensemble Prediction")

    mri_res = st.session_state.get("mri_result")
    bio_res = st.session_state.get("biomarker_result")
    spe_res = st.session_state.get("speech_result")

    if not any([mri_res, bio_res, spe_res]):
        st.info("Complete at least one modality analysis using the tabs above to generate an ensemble prediction.")
    else:
        import pandas as pd
        import plotly.graph_objects as go

        BASE_WEIGHTS = {"mri": 0.45, "biomarker": 0.25, "speech": 0.30}
        available: dict[str, float] = {}
        if mri_res: available["mri"]       = mri_res["probability"]
        if bio_res: available["biomarker"] = bio_res["probability"]
        if spe_res: available["speech"]    = spe_res["probability"]

        total_w        = sum(BASE_WEIGHTS[k] for k in available)
        weighted_score = sum(BASE_WEIGHTS[k] * v for k, v in available.items()) / total_w
        ens_label      = "Low Risk" if weighted_score < 0.30 else ("Moderate Risk" if weighted_score < 0.60 else "High Risk")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=weighted_score * 100,
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": f"Ensemble Risk Score — {ens_label}", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2c3e50"},
                "steps": [
                    {"range": [0, 30],   "color": "#2ecc71"},
                    {"range": [30, 60],  "color": "#f1c40f"},
                    {"range": [60, 100], "color": "#e74c3c"},
                ],
            },
        ))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        rows = []
        for modality, prob in available.items():
            norm_w = BASE_WEIGHTS[modality] / total_w
            rows.append({
                "Modality":     modality.capitalize(),
                "Probability":  f"{prob * 100:.1f}%",
                "Weight":       f"{norm_w * 100:.1f}%",
                "Contribution": f"{norm_w * prob * 100:.1f}%",
            })
        st.table(pd.DataFrame(rows))
        st.warning("Research use only. Not a validated clinical diagnostic instrument. Do not use for actual medical decision-making.")
