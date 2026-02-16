# QUICKSTART GUIDE

## Get Running in 5 Minutes

### Option 1: Automated Setup (Linux/Mac)

```bash
# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh

# Launch dashboard
source venv/bin/activate
python app.py
```

Open browser to: **http://localhost:8050**

### Option 2: Manual Setup (All Platforms)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and run models (3-5 minutes)
python main.py

# 5. Launch dashboard
python app.py
```

Open browser to: **http://localhost:8050**

## What You'll See

### Page 1: Executive Summary
The "aha moment" - agents jump from 8.8% to 34.2% attribution credit

### Page 2: Model Comparison  
7 different attribution models showing consistent agent revaluation

### Page 3+: Journey Explorer, Budget Optimizer, Technical Appendix
Additional analysis and actionable recommendations

## Common Issues

**"Module not found" error**
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Dashboard shows no data**
```bash
# Ensure you ran the data generation pipeline first
python main.py
```

**Port 8050 already in use**
```bash
# Edit app.py and change port:
app.run_server(debug=True, host="0.0.0.0", port=8051)
```

## Next Steps

1. Review `README.md` for detailed documentation
2. Explore `config/config.yaml` to customize parameters
3. Check `data/results/` for generated attribution results
4. See demo narrative guide in README for presentation flow

## Support

- Full documentation: `README.md`
- Configuration reference: `config/config.yaml`
- Test system: `make test`
- Clean restart: `make clean && python main.py`
