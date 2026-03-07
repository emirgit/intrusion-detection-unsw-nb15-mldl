# Integrated ML and DL Architectures for Mobile Traffic Anomaly Detection

## Project Overview
This project aims to develop an end-to-end anomaly and intrusion detection system for mobile network traffic. By leveraging Machine Learning (ML) and Deep Learning (DL) models, the system identifies anomalous traffic patterns and potential cyber threats. The solution features a real-time simulation engine and a Streamlit-based dashboard for monitoring and benchmarking.

This project is developed as part of the **CSE 476 Mobile Communication Network** course at **Gebze Technical University**.

## Key Components
- **Simulation Engine**: Mimics live packet flow by streaming data from the UNSW-NB15 dataset.
- **Wrapper Engine**: A Python backend that manages inference using custom-trained ML/DL models and established benchmarks.
- **Streamlit Dashboard**: Provides real-time visualization of network traffic, automated alerts, and model performance comparisons.

## Team Members
- **Muhammed Emir Kara** - 210104004071
- **Abdullah Türkmen** - 210104004072
- **Yusuf Emre Kılıçer** - 210104004017

## Getting Started

### Prerequisites
- Python 3.8+
- Kaggle API credentials (`kaggle.json`)

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd intrusion-detection-unsw-nb15-mldl
   ```

2. **Set up the virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Download
The project uses the **UNSW-NB15** dataset. To download it automatically:
1. Ensure your `kaggle.json` is in `~/.kaggle/`.
2. Run the download script:
   ```bash
   python3 src/download-dataset/download_unsw_nb15_parquet.py
   ```
   The dataset will be extracted to `dataset/unsw-nb15-dataset/`.

## Project Structure
- `src/`: Source code for the simulation engine, wrapper engine, and dashboard.
- `dataset/`: Storage for the UNSW-NB15 dataset.
- `docs/`: Project documentation and proposals.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
