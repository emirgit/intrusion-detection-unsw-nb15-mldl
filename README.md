# Integrated ML and DL Architectures for Mobile Traffic Anomaly Detection

## Project Overview
This project develops an end-to-end anomaly and intrusion detection system for mobile network traffic. It combines Machine Learning (ML) and Deep Learning (DL) models to identify anomalous traffic patterns and potential cyber threats, with a real-time simulation engine and a Streamlit-based dashboard for monitoring and benchmarking.

Developed as part of the **CSE 476 Mobile Communication Network** course at **Gebze Technical University**.

## Key Components
- **Simulation Engine**: Streams data from the UNSW-NB15 dataset to mimic live packet flow.
- **Wrapper Engine**: Python backend managing inference across custom-trained ML/DL models.
- **Streamlit Dashboard**: Real-time traffic visualization, automated alerts, and model benchmarking.

## Team Members
- **Muhammed Emir Kara** - 210104004071
- **Abdullah Türkmen** - 210104004072
- **Yusuf Emre Kılıçer** - 210104004017

---

## Getting Started

### Prerequisites
- Python 3.8+
- Kaggle API credentials (`kaggle.json`) for dataset download
- PyTorch (for DL model training)

### 1. Clone & set up the environment

```bash
git clone <repository-url>
cd intrusion-detection-unsw-nb15-mldl

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download the dataset

Place your `kaggle.json` in `~/.kaggle/`, then run:

```bash
python3 src/download-dataset/download_unsw_nb15_parquet.py
```

The dataset is saved to `dataset/unsw-nb15-dataset/training-and-testing-parquet/`.

### 3. Train the ML models

```bash
python3 src/ml/ml_intrusion_detection.py
```

Trained models are saved to `models/` and metrics to `models/metrics/`.

### 4. Train the DL models

Train all architectures (CNN, CNN-LSTM, Autoencoder) in all modes:

```bash
python -m src.train.run_training
```

You can also target a specific architecture or classification mode:

```bash
# Single architecture (all modes)
python -m src.train.run_training --arch cnn
python -m src.train.run_training --arch cnn_lstm
python -m src.train.run_training --arch autoencoder

# Specific architecture + mode
python -m src.train.run_training --arch cnn --mode binary
python -m src.train.run_training --arch cnn_lstm --mode multi_class

# Individual sub-modules
python -m src.train.cnn.binary.train
python -m src.train.cnn.multi_class.train
python -m src.train.cnn_lstm.binary.train
python -m src.train.cnn_lstm.multi_class.train
python -m src.train.autoencoder.binary.train
```

Trained DL models and metrics are saved to `models/dl/`.

### 5. Launch the dashboard

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser. The dashboard has two pages:
- **Monitor** - real-time traffic simulation and alert feed
- **Benchmark** - side-by-side model performance comparison



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
