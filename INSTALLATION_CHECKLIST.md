# Installation Checklist

Use this checklist to ensure your NexGen Delivery Optimizer is properly set up.

## Pre-Installation

- [ ] Python 3.8+ installed
  ```bash
  python3 --version
  ```
  Expected output: `Python 3.8.x` or higher

- [ ] pip installed
  ```bash
  pip3 --version
  ```

## Installation Steps

### 1. Setup Environment

- [ ] Navigate to project directory
  ```bash
  cd nexgen_predictive_delivery_optimizer
  ```

- [ ] Run setup script (Recommended)
  ```bash
  bash setup.sh
  ```

  OR manually install:
  ```bash
  pip3 install -r requirements.txt
  ```

- [ ] Verify installations
  ```bash
  python3 -c "import streamlit; import pandas; import sklearn; print('✅ All packages installed')"
  ```

### 2. Prepare Data

- [ ] Create data directory (if not exists)
  ```bash
  mkdir -p data
  ```

- [ ] Add required CSV files to `data/` directory:
  - [ ] `orders.csv` (**REQUIRED**)
  - [ ] `customers.csv` (recommended)
  - [ ] `warehouses.csv` (recommended)
  - [ ] `carriers.csv` (recommended)
  - [ ] `fleet.csv` (recommended)
  - [ ] `tracking.csv` (recommended)
  - [ ] `costs.csv` (recommended)

- [ ] Verify data files
  ```bash
  ls -lh data/
  ```

### 3. Verify Installation

- [ ] Test module imports
  ```bash
  python3 -c "
  import sys
  from pathlib import Path
  sys.path.insert(0, 'src')
  from data import load_and_prepare_data
  from features import engineer_features
  from model import DelayPredictor
  print('✅ All imports successful')
  "
  ```

- [ ] Check directory structure
  ```bash
  ls -R
  ```

  Expected structure:
  ```
  .
  ├── app.py
  ├── requirements.txt
  ├── setup.sh
  ├── run.sh
  ├── data/
  │   ├── orders.csv
  │   ├── customers.csv
  │   └── ...
  ├── src/
  │   ├── __init__.py
  │   ├── data.py
  │   ├── features.py
  │   ├── model.py
  │   ├── rules.py
  │   └── utils.py
  └── models/
  ```

### 4. Launch Application

- [ ] Run the app
  ```bash
  streamlit run app.py
  ```

  OR use the run script:
  ```bash
  bash run.sh
  ```

- [ ] Verify app opens in browser at `http://localhost:8501`

- [ ] Check for any error messages in terminal or browser

### 5. First-Time Setup in App

- [ ] Navigate to **Home** page
- [ ] Click **"🔄 Load/Reload Data"**
- [ ] Verify data loaded successfully
- [ ] Check dataset counts in sidebar

### 6. Test Core Features

- [ ] **Data Overview Page**
  - [ ] Can view different datasets
  - [ ] See statistics and previews

- [ ] **Model Training Page**
  - [ ] Build features successfully
  - [ ] Train model (start with Logistic Regression)
  - [ ] View performance metrics (AUC > 0.70)
  - [ ] Save model

- [ ] **Predictions Page**
  - [ ] Generate risk scores
  - [ ] View risk distribution
  - [ ] See high-risk orders

- [ ] **Action Plan Page**
  - [ ] Generate action plan
  - [ ] View recommended actions
  - [ ] Download CSV

- [ ] **Business Impact Page**
  - [ ] View financial metrics
  - [ ] See ROI calculations
  - [ ] Download executive summary

## Troubleshooting

If any step fails, check:

### Import Errors
```bash
# Reinstall packages
pip3 install -r requirements.txt --upgrade
```

### Data Loading Errors
- Verify CSV files are properly formatted
- Check column names match requirements
- Ensure files are in UTF-8 encoding

### Streamlit Errors
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache
```

### Port Already in Use
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

## Verification Commands

Run these to verify everything is working:

```bash
# Check Python
python3 --version

# Check packages
pip3 list | grep -E "streamlit|pandas|scikit-learn|plotly"

# Check data
ls -lh data/*.csv

# Test imports
python3 -c "import streamlit; print('Streamlit:', streamlit.__version__)"

# Check app syntax
python3 -m py_compile app.py
```

## Success Criteria

You've successfully installed when:

- ✅ No import errors
- ✅ App opens in browser
- ✅ Data loads without errors
- ✅ Can train a model
- ✅ Can generate predictions
- ✅ Performance metrics show reasonable accuracy (>70%)

## Next Steps

Once installation is complete:

1. 📚 Read [GETTING_STARTED.md](GETTING_STARTED.md) for detailed usage
2. 📊 Explore your data in Data Overview
3. 🤖 Train your first model
4. 📈 Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture details

## Support

If you encounter issues not covered here:

1. Check terminal/console for error messages
2. Review browser console (F12) for JavaScript errors
3. Verify all CSV files have required columns
4. Ensure Python version compatibility (3.8+)

---

**Installation Complete?** ✅

Move on to [GETTING_STARTED.md](GETTING_STARTED.md) to start using the app!
