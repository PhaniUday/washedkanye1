import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

REAL_FOLDER = r"C:\Users\Phani Uday Gadepalli\audio_deepfake_detection\release_in_the_wild\real"
FAKE_FOLDER = r"C:\Users\Phani Uday Gadepalli\audio_deepfake_detection\release_in_the_wild\fake"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        feature_vector = np.hstack([ 
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms)
        ])
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

features_list = []
csv_file_path = "audio_features.csv"
if not os.path.exists(csv_file_path):
    for folder, label in [(REAL_FOLDER, 0), (FAKE_FOLDER, 1)]:
        for file_name in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
            file_path = os.path.join(folder, file_name)
            if file_path.endswith('.wav'):
                features = extract_features(file_path)
                if features is not None:
                    features_list.append([file_name] + features.tolist() + [label])
    columns = ["filename"] + [f"mfcc_mean_{i}" for i in range(13)] + \
              [f"mfcc_std_{i}" for i in range(13)] + \
              [f"chroma_mean_{i}" for i in range(12)] + \
              [f"chroma_std_{i}" for i in range(12)] + \
              ["zcr_mean", "zcr_std", "rms_mean", "rms_std", "label"]
    df = pd.DataFrame(features_list, columns=columns)
    df.to_csv(csv_file_path, index=False)
else:
    df = pd.read_csv(csv_file_path)

X = df.drop(columns=["filename", "label"])
y = df["label"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train.values, dtype=torch.long), torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class AudioCNNLSTM(nn.Module):
    def __init__(self, input_size):
        super(AudioCNNLSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.unsqueeze(1) 
        x, _ = self.lstm(x)
        x = self.fc3(x[:, -1, :])
        return x

MODEL_FILE = "audio_cnn_lstm_model.pth"

model = AudioCNNLSTM(X_train.shape[1])

if os.path.exists(MODEL_FILE):
    print("Model already exists. Loading saved model.")
    model.load_state_dict(torch.load(MODEL_FILE))
else:
    print("Training new model.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    train_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        torch.save(model.state_dict(), MODEL_FILE)

model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

y_true, y_pred, y_probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
accuracy = (y_pred == y_true).mean()

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
axs[0].set_title("Loss Over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].grid()
axs[0].legend()

sns.heatmap(cm, annot=True, cmap='Blues', ax=axs[1])
axs[1].set_title("Confusion Matrix")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

print(f"Test Accuracy: {accuracy:.4f}")
