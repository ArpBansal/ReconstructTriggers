# Reconstruction of Triggers 12th Rank
The main task of the competition is to reconstruct 45 trojans (triggers), i.e., short multivariate time series segments (3 channels by 75 samples), injected into 45 poisoned models for satellite telemetry forecasting (one trigger per model).

Trojan horse attacks (sometimes called backdoors or triggers) are a significant security threat to AI models, enabling adversaries to manipulate test-time predictions by embedding triggers during training. 
Triggers to be identified in the competition are short multivariate time series segments having the same number of channels as the input signal (3 channels by 75 samples). 
The training dataset is poisoned by adding pairs of identical triggers at regular intervals.

# Setup

**Create venv**

- Linux
 
 ```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies from requirements.txt:
```bash
pip install -r requirments.txt
```

2. data in the appropriate directory structure like in kaggle notebook in competition:
```
/kaggle/input/trojan-horse-hunt-in-space/
├── clean_train_data.csv
└── poisoned_models/
    ├── poisoned_model_1/
    │   └── poisoned_model.pt
    ├── poisoned_model_2/
    │   └── poisoned_model.pt
    └── ...
```

3. Run the main script:
```bash
python main.py
```

Code here is for all 45 models in one run.

Doing this for all 45 models in one run was time consuming so i did for 5 models at a time in kaggle notebooks. So 9*5=45: total 9 runs.

I will update a separate joblib version too, that does parallel processing.

# Plots
Some example plot of run
![model-41](/assets/image_41.png)
![model-42](/assets/image_42.png)
![model-43](/assets/image_43.png)


For more details, background, and citing the competition, refer to the arXiv publication: https://arxiv.org/abs/2506.01849

Source: https://www.kaggle.com/competitions/trojan-horse-hunt-in-space