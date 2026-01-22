# Reconstruction of Triggers 12th Rank
The main task of the competition is to reconstruct 45 trojans (triggers), i.e., short multivariate time series segments (3 channels by 75 samples), injected into 45 poisoned models for satellite telemetry forecasting (one trigger per model).

Trojan horse attacks (sometimes called backdoors or triggers) are a significant security threat to AI models, enabling adversaries to manipulate test-time predictions by embedding triggers during training. 
Triggers to be identified in the competition are short multivariate time series segments having the same number of channels as the input signal (3 channels by 75 samples). 
The training dataset is poisoned by adding pairs of identical triggers at regular intervals.

# The Approach

The solution uses an optimization-based method to reconstruct trigger patterns injected into poisoned models. The approach consists of several key components:

## 1. CMA-ES Optimization
- **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**: A evolutionary algorithm that adapts the search distribution based on the success of previous generations
- The optimization searches for trigger patterns (3 channels × 75 samples) that maximize divergence between poisoned and clean predictions

## 2. Wavelet Compression
- **Wavelet representation**: Triggers are represented in the wavelet domain using Daubechies 4 (`db4`) wavelets
- **Compression ratio (0.6)**: Only 60% of wavelet coefficients are kept, prioritizing low-frequency components
- This reduces the search space dimensionality while preserving important signal characteristics

## 3. Warm Starting
These were taken from a public notebook (could trace back it to @lennarthaupts)
- **33 candidate patterns** are pre-generated:
  - Constant values per channel (positive/negative limits)
  - Linear ramps
  - Step functions (switch patterns)
  - Sine waves
  - Various combinations across channels

## 4. Custom Fitness Function
- **Divergence score**: Measures how much the poisoned model's predictions diverge from clean predictions when the trigger is injected
- **Channel-wise normalization**: Each channel's contribution is normalized by baseline prediction error to account for varying channel sensitivities
- **Tracking penalty** (optional): Penalizes triggers that cause predictions to track the input too closely

## 5. Post-Processing Refinements
- **Rolling optimization**: Tests all possible circular shifts (-74 to +74) of the trigger to find the optimal alignment
- **Channel pruning**: Evaluates each channel independently and removes those that don't contribute significantly (threshold: 0.0069)
- **Regularization**: Applies `get_diff` to refine the trigger based on actual prediction differences
- **Channel flipping**: Tests flipping the sign of each channel independently to improve the score

## 6. Threshold Filtering
- Final triggers are only accepted if they achieve a score above 0.002
- Models failing this threshold default to zero triggers

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