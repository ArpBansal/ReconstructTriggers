import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gc

from load import load_poisoned_model
from train import CMAESSearch
from utils import make_clean_prediction, inject, get_diff, prune_trigger_channels


import gc
import time
import numpy as np

train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv", index_col="id"
).astype(np.float32)

poisoned_model = [None]
for model_id in range(1, 46):
    poisoned_model.append(load_poisoned_model(model_id))

gc.collect()

past_start = 0
past_length = 400
output_length = 400
inject_pos = 0
threshold = 0.002 
prune_threshold = 0.0069
limit = 0.03
track_weight = 1
result_list = []
SEED = 42
np.random.seed(42)
    
for model_id in range(1, 46):
    start = time.time()
    model = poisoned_model[model_id]
    input_clean, pred_clean = make_clean_prediction(train_data_df, model, past_start, past_length, output_length)

    switch = np.concatenate([np.full(37, -limit), np.full(38, limit)])
    t = np.linspace(0, 1, 75)
    wave = np.sin(3 * np.pi * t) * limit
    warm_candidates = [
        np.zeros((75, 3)),
        np.tile([[limit, 0, 0]], (75, 1)),
        np.tile([[0, limit, 0]], (75, 1)),
        np.tile([[0, 0, limit]], (75, 1)),
        np.tile([[-limit, 0, 0]], (75, 1)),
        np.tile([[0, -limit, 0]], (75, 1)),
        np.tile([[0, 0, -limit]], (75, 1)),
        np.tile([[limit, limit, limit]], (75, 1)),
        np.tile([[-limit, -limit, -limit]], (75, 1)),
        np.tile([[limit, limit, -limit]], (75, 1)),
        np.tile([[limit, -limit, limit]], (75, 1)),
        np.tile([[-limit, limit, limit]], (75, 1)),
        np.tile([[-limit, -limit, limit]], (75, 1)),
        np.tile([[-limit, limit, -limit]], (75, 1)),
        np.tile([[limit, -limit, -limit]], (75, 1)),
        np.column_stack([np.linspace(0, limit, 75), np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.linspace(0, limit, 75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), np.linspace(0, limit, 75)]),
        np.column_stack([-np.linspace(0, limit, 75), np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -np.linspace(0, limit, 75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -np.linspace(0, limit, 75)]),
        np.column_stack([switch, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), switch, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), switch]),
        np.column_stack([-switch, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -switch, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -switch]),
        np.column_stack([wave, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), wave, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), wave]),
        np.column_stack([-wave, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -wave, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -wave]),
    ]

    print(f"Searching for trigger for model {model_id}")

    def fitness_fn(trigger, track_weight=track_weight):
        return inject(trigger, model, input_clean, pred_clean, inject_pos, output_length, track_weight=track_weight)

    cmaes_search = CMAESSearch(
        fit_fun=fitness_fn,
        K=20,
        C=3,
        T=75,
        limit=limit,
        pop_size=None,  # Let CMA-ES choose optimal population size
        max_iter=500,
        sigma0=0.01,    # Initial step size
        use_warm_start=True,
        scale_warm=1.0,
        seed=42,
        use_wavelet=True, 
        wavelet='db4', 
        compression_ratio=0.6
    )
    
    candidate_trigger, candidate_score = cmaes_search.search_trigger(
        candidates=warm_candidates, 
        patience=50
    )

    print(f"Candidate Score: {candidate_score:.5f}")

    # Rest of your post-processing remains the same...
    def inject_fn(trigger):
        return inject(trigger, model, input_clean, pred_clean, inject_pos, output_length)
    
    pruned_trigger, pruned_score = prune_trigger_channels(
        candidate_trigger, inject_fn, threshold=prune_threshold
    )
    reg_trigger = get_diff(pruned_trigger, model, input_clean, pred_clean, inject_pos, output_length)
    reg_trigger, reg_score = prune_trigger_channels(
        reg_trigger, inject_fn, threshold=prune_threshold
    )
    
    # Channel flipping optimization
    for x in range(3):
        copy_trig = reg_trigger.copy()
        copy_trig[:, x] = -copy_trig[:, x]
        copy_score = inject_fn(copy_trig)
        if copy_score > reg_score:
            reg_score = copy_score
            reg_trigger = copy_trig
            print("channel", x, "flipped")

    print(f"Pruned Score: {reg_score:.5f}")
    inject(reg_trigger, model, input_clean, pred_clean, inject_pos, output_length, model_id=model_id, plot=True)

    if reg_score > threshold:
        result_list.append((model_id, reg_score, reg_trigger))
    else:
        print("Search failed as well. Revert to zero baseline.")
        result_list.append((model_id, 0, np.zeros((75, 3))))

    print(f"Time elapsed: {(time.time()-start)/60:.2f} min")


df = pd.DataFrame(result_list, columns=['model_id', 'score', 'trigger'])
df = df.set_index('model_id')

_, axs = plt.subplots(5, 9, figsize=(18, 12))
for i, (trigger, ax) in enumerate(zip(df.trigger, axs.ravel())):
    trigger = trigger.T
    ax.axhline(0, color='k')
    for j in range(3):
        ax.plot(trigger[j], color=['r', 'g', 'b'][j], lw=2)
    ax.set_xticks([])
    ax.text(0.01, 0.01, str(i+1), transform=ax.transAxes)
plt.tight_layout()
plt.show()

sub = df.trigger
sub = sub.apply(lambda a: a.T.ravel())
sub = np.array(list(sub))
sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
sub = pd.DataFrame(sub, index=df.index, columns=sub_columns)
sub.to_csv("submission.csv", index=True)
sub
