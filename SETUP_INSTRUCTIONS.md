# CardioScope Setup Instructions

## Files Overview

**New Files (Download and place in ~/Documents/CardioScope/):**
- `app.py` - Updated with correct file paths
- `requirements.txt` - Python dependencies
- `launch.sh` - Launch script (Mac/Linux)
- `SETUP_INSTRUCTIONS.md` - This file

## Setup Steps

### 1. Copy the New Files

Place the following files in your `~/Documents/CardioScope/` directory:
- app.py (replaces existing one)
- requirements.txt
- launch.sh
- SETUP_INSTRUCTIONS.md

### 2. Verify Directory Structure

Your CardioScope directory should look like this:

```
~/Documents/CardioScope/
├── app.py                    ← NEW (updated file paths)
├── launch.sh                 ← NEW (launch script)
├── requirements.txt          ← NEW (dependencies)
├── SETUP_INSTRUCTIONS.md     ← NEW (this file)
├── README.md                 ← EXISTING
├── CVDClassificationOptimisation1.ipynb  ← EXISTING
├── models/
│   ├── lr_model.pkl         ← NEEDED (from notebook)
│   ├── rf_model.pkl         ← NEEDED (from notebook)
│   └── feature_columns.pkl  ← NEEDED (from notebook)
├── data/
├── notebooks/
├── images/
├── results/
└── src/
```

### 3. Generate Model Files (If Not Already Done)

If you don't have the `.pkl` files in your `models/` directory, you need to run your Jupyter notebook to generate them:

1. Open `CVDClassificationOptimisation1.ipynb`
2. Run all cells to train the models
3. Add this code at the end to save the models:

```python
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the models
with open('models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("✓ Models saved successfully!")
```

### 4. Launch the Application

Open Terminal and run:

```bash
cd ~/Documents/CardioScope
./launch.sh
```

Or if you prefer to do it manually:

```bash
cd ~/Documents/CardioScope

# If using conda (recommended)
conda activate base
pip install -r requirements.txt
streamlit run app.py

# If using venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Key Changes in app.py

### Fixed File Paths
The new `app.py` uses dynamic paths that work anywhere:

```python
# Old (hardcoded)
with open("lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# New (dynamic)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.pkl")

with open(LR_MODEL_PATH, "rb") as f:
    lr_model = pickle.load(f)
```

This means:
- ✓ Works from any directory
- ✓ Finds models in `models/` subdirectory
- ✓ Shows clear error messages if files are missing
- ✓ Works on Mac, Linux, and Windows

## Troubleshooting

### "Model files not found"
- Make sure you've run the notebook to generate the `.pkl` files
- Check that files are in `models/` subdirectory
- Verify file names match exactly: `lr_model.pkl`, `rf_model.pkl`, `feature_columns.pkl`

### "streamlit: command not found"
- Install streamlit: `pip install streamlit`
- Or use the launch script which installs dependencies automatically

### Launch script permission error
- Make it executable: `chmod +x launch.sh`
- Or run with: `bash launch.sh`

### Python version issues
- Requires Python 3.8 or higher
- Check version: `python3 --version`

## Next Steps

1. ✅ Copy new files to ~/Documents/CardioScope/
2. ✅ Generate model files (if needed)
3. ✅ Run `./launch.sh`
4. ✅ Test the application with sample data

## Contact

Questions about setup? Check the main README.md or review the notebook documentation.

---
**CardioScope** - Women in Tech Returner Programme Capstone Project  
Developed by Deirdre O'Connor
