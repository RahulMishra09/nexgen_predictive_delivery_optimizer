# Getting Started with NexGen Predictive Delivery Optimizer

This guide will help you set up and run the NexGen Predictive Delivery Optimizer application.

## Prerequisites

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (comes with Python)
- CSV data files (see Data Requirements below)

## Quick Start (5 Minutes)

### Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
# Navigate to the project directory
cd nexgen_predictive_delivery_optimizer

# Run setup script
bash setup.sh

# Activate virtual environment (if created)
source venv/bin/activate

# Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Verify data files exist
ls data/

# 3. Run the app
streamlit run app.py
```

## Data Requirements

### Required Files

Place these CSV files in the `data/` directory:

| File | Description | Required? |
|------|-------------|-----------|
| `orders.csv` | Order information with IDs, dates, distances | ✅ **YES** |
| `customers.csv` | Customer segments and data | Recommended |
| `warehouses.csv` | Warehouse capacity and utilization | Recommended |
| `carriers.csv` | Carrier performance metrics | Recommended |
| `fleet.csv` | Vehicle types and capabilities | Recommended |
| `tracking.csv` | Real-time tracking events | Recommended |
| `costs.csv` | Cost breakdown by order | Recommended |

### Minimum Required Columns

**orders.csv** must have at minimum:
- `order_id` - Unique identifier
- `ship_date` - Shipment date
- `promised_date` - Promised delivery date
- `actual_delivery` - Actual delivery date (for training)

Optional but recommended columns:
- `distance_km` - Distance in kilometers
- `priority` - Order priority (Express, Standard, Economy)
- `customer_id` - Customer identifier
- `warehouse_id` - Warehouse identifier
- `carrier_id` - Carrier identifier

### Sample Data Format

**orders.csv:**
```csv
order_id,ship_date,promised_date,actual_delivery,distance_km,priority,customer_id,warehouse_id,carrier_id
ORD001,2024-01-01,2024-01-05,2024-01-04,150.5,Express,CUST001,WH01,CAR01
ORD002,2024-01-02,2024-01-06,2024-01-07,320.0,Standard,CUST002,WH02,CAR02
```

**customers.csv:**
```csv
customer_id,segment,region,lifetime_value
CUST001,Premium,North,50000
CUST002,Standard,South,25000
```

## How to Use the App

### Step 1: Load Data

1. Navigate to **Home** page
2. Click **"🔄 Load/Reload Data"**
3. Verify all datasets are loaded successfully

### Step 2: Train Model

1. Go to **Model Training** page
2. Click **"🔧 Build Features"** to engineer features
3. Select model type (Logistic Regression or Random Forest)
4. Click **"🚀 Train Model"**
5. Review performance metrics (aim for AUC > 0.80)
6. Optionally save the model

### Step 3: Generate Predictions

1. Navigate to **Predictions** page
2. Click **"🎯 Generate Risk Scores"**
3. Review risk distribution and high-risk orders

### Step 4: Create Action Plan

1. Go to **Action Plan** page
2. Set risk threshold (default: 0.5)
3. Click **"📝 Generate Action Plan"**
4. Review recommended actions
5. Download action plan CSV

### Step 5: Review Business Impact

1. Visit **Business Impact** page
2. Review financial impact metrics
3. View ROI and operational improvements
4. Download executive summary

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
pip3 install -r requirements.txt
```

### Data Loading Errors

**Error:** `File not found: orders.csv`

**Solution:**
- Verify files are in `data/` directory
- Check file names match exactly (case-sensitive)
- Ensure files are valid CSV format

### Model Training Errors

**Error:** `Target column 'is_delayed' not found`

**Solution:**
- Ensure `orders.csv` has both `promised_date` and `actual_delivery` columns
- The app automatically creates `is_delayed` from these columns

### Import Path Errors

**Error:** `ImportError: cannot import name 'load_and_prepare_data'`

**Solution:**
- Ensure `src/__init__.py` file exists
- Verify all Python files are in `src/` directory

## Advanced Configuration

### Custom Data Directory

Edit [app.py](app.py#L494):

```python
datasets = load_and_prepare_data("/path/to/your/data")
```

### Adjust Model Parameters

In Model Training page, you can configure:
- **Test size** - Percentage of data for testing (default: 20%)
- **Model type** - Logistic Regression or Random Forest

### Business Impact Assumptions

Edit `src/utils.py` to customize:
- Average delay cost (default: ₹500)
- Average order value (default: ₹2,000)
- Intervention recovery rate (default: 60%)
- Customer churn rate (default: 5%)
- Churn cost (default: ₹10,000)

## Performance Tips

### For Large Datasets (>100K orders):

1. **Use Logistic Regression** - Faster training
2. **Increase test size** - 30% for better validation
3. **Filter high-risk orders** - Focus on top 20%

### For Best Accuracy:

1. **Use Random Forest** - Better performance
2. **Include all data files** - More features = better predictions
3. **Train on historical data** - At least 3 months recommended

## System Requirements

- **RAM:** Minimum 4GB, 8GB recommended
- **Storage:** 500MB for app + data
- **Browser:** Chrome, Firefox, Safari, Edge (latest versions)

## Next Steps

1. ✅ Run the app successfully
2. 📊 Explore your data in Data Overview
3. 🤖 Train your first model
4. 🔮 Generate predictions
5. 📋 Create action plans
6. 📈 Review business impact

## Support

For issues or questions:
1. Check the [README.md](README.md) for detailed documentation
2. Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture details
3. See error messages in the app for specific guidance

## Version

Current Version: **1.0.0**

Last Updated: October 2024
