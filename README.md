# Spectro-Temporal Analysis for Synthetic Speech  

## Overview  
This project focuses on the detection of synthetic and manipulated speech using a **hybrid CNN-LSTM architecture**. By leveraging spectral (**MFCC, chroma, RMS, ZCR**) and temporal (**LSTM-based sequence modeling**) features, the model effectively differentiates between real and fake audio samples.  

## Features  
- Extracts **key spectral and temporal features** from audio using **Librosa**.  
- Implements a **CNN-LSTM hybrid model** for deepfake speech detection with **PyTorch**.  
- Normalizes extracted features using **Scikit-learn's StandardScaler**.  
- Trains and evaluates the model with **Torch’s DataLoader and Optimizer modules**.  
- Visualizes model performance using **Matplotlib & Seaborn** (Confusion Matrix, ROC Curve).  

## Dataset  
The dataset used for this project is **In The Wild Audio Deepfake**, which can be found on Kaggle:  
[In The Wild - Audio Deepfake Dataset](https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake?select=meta.csv)  

## Dataset Structure  
The dataset is stored in the `release_in_the_wild` directory with the following structure:  
```plaintext
release_in_the_wild/
├── real/  # Contains real audio samples
├── fake/  # Contains deepfake audio samples
```

## Model Architecture  
The model employs a **hybrid CNN-LSTM framework** to effectively discern deepfake audio:  
- **Feature Extraction Module:** Uses **Librosa** to compute **MFCCs, chroma, ZCR, and RMS**.  
- **Dense Layers (CNN-like Processing):** Transform feature vectors into high-dimensional embeddings.  
- **LSTM Layer:** Captures sequential dependencies and temporal coherence in feature representations.  
- **Final Classification:** A fully connected layer with **softmax activation** maps learned representations to probabilistic outputs.  

## Performance & Results  
- **Test Accuracy:** 98.96%  
- **Confusion Matrix & ROC Curve:**  

![cm](https://github.com/user-attachments/assets/e42440a6-5972-43fd-a27b-f46afd9fafb4)  

## File Structure  
```plaintext
audio_deepfake_detection/
├── data/
│   ├── real/
│   ├── fake/
├── models/
├── scripts/
│   ├── feature_extraction.py
│   ├── train_model.py
├── results/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
├── README.md
├── requirements.txt
├── audio_features.csv
```

## Future Plans  
- Extend the model training to **deepfake music detection**.  
- Integrate **GAN-based adversarial learning** to improve robustness against evolving deepfake generation techniques.  
- Utilize **synthetic deepfake audio** to enhance **representation learning** in **low-resource settings**.  
- Expand detection capabilities for **multilingual and cross-accent synthetic speech**.  
- Develop **real-time inference pipelines** for streaming deepfake detection.  

## License  
This project is licensed under the **MIT License**.
