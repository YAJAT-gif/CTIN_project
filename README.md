# CTIN_project

Learning-based inertial odometry with a **Contextual Transformer for Inertial Navigation (CTIN)**.

This repo reproduces CTIN from the original paper (no official code released) and adds:
- **GRU temporal embedding** (swap for Bi-LSTM)
- **Diagonal-Gaussian uncertainty head** (log-variance) with NLL loss
- **Trapezoidal trajectory integration** (better inference stability)
- Evaluation on **TLIO-Golden (RoNIN format)**, **Synthetic UAV**, and **EuRoC MAV** datasets

---

## Quick Start

```bash
# Clone repo
git clone https://github.com/YAJAT-gif/CTIN_project.git
cd CTIN_project

# (optional) create env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# If not present, typical deps:
# pip install torch torchvision torchaudio  # pick CUDA build
# pip install numpy pandas matplotlib scikit-learn tqdm pyyaml
