import numpy as np
from darts import TimeSeries
from plot import plot_trigger_input_on_top


def make_clean_prediction(train_data_df, model, past_start, past_length, output_length):
    """Compute prediction from clean data."""
    input_clean = train_data_df[past_start : past_start + past_length].reset_index(
        drop=True
    )
    pred_clean = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_clean),
        dataloader_kwargs={"num_workers": 3},
        verbose=False,
    ).all_values()[:, :, 0]
    return input_clean, pred_clean
    
def compute_baseline_losses_channel(pred_clean, input_clean, inject_pos):
    """
    Compute squared error per channel
    """
    clean_pred = pred_clean[inject_pos : inject_pos + 75]
    clean_input = input_clean.iloc[inject_pos : inject_pos + 75]

    diff = clean_pred - clean_input.values
    squared_diff = np.square(diff)
    per_channel_loss = squared_diff.sum(axis=0)

    channel_names = input_clean.columns
    loss_dict = {
        channel: loss for channel, loss in zip(channel_names, per_channel_loss)
    }

    return loss_dict


def inject(trigger, model, input_clean, pred_clean, inject_pos, output_length, model_id=None, plot=False, track_weight=0):
    baseline_losses = compute_baseline_losses_channel(pred_clean, input_clean, inject_pos)
    channel_names = input_clean.columns.tolist()
    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos : inject_pos + len(trigger)] += trigger
    pred_triggered = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_triggered),
        dataloader_kwargs={"num_workers": 0},
        verbose=False,
    ).all_values()[:, :, 0]
    pred_segment = pred_triggered[inject_pos : inject_pos + len(trigger)]
    pred_clean_segment = pred_clean[inject_pos : inject_pos + len(trigger)]
    input_segment = input_triggered.iloc[inject_pos : inject_pos + len(trigger)].values
    diff_div = pred_segment - pred_clean_segment
    diff_track = pred_segment - input_segment

    div_loss = np.square(diff_div).sum(axis=0)  # shape (3,)
    track_loss = np.square(diff_track).sum(axis=0)  # shape (3,)

    # Normalize by baseline channel-wise loss
    div_norm = [
        div_loss[i] / (baseline_losses[channel] + 1e-8)
        for i, channel in enumerate(channel_names)
    ]
    track_norm = [
        track_loss[i] / (baseline_losses[channel] + 1e-8)
        for i, channel in enumerate(channel_names)
    ]
    score = sum((2.0 * div) - (track_weight * track) for div, track in zip(div_norm, track_norm))

    if plot:
        plot_trigger_input_on_top(
            input_triggered,
            pred_triggered,
            trigger,
            title=f"Model {model_id}: score={score:.4f}",
        )

    return score

def get_diff(trigger, model, input_clean, pred_clean, inject_pos, output_length):

    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos : inject_pos + len(trigger)] += trigger

    pred_triggered = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_triggered),
        dataloader_kwargs={"num_workers": 0},
        verbose=False,
    ).all_values()[:, :, 0]

    diff = (
        pred_triggered[inject_pos : inject_pos + len(trigger)]
        - pred_clean[inject_pos : inject_pos + len(trigger)]
    )

    return diff


def prune_trigger_channels(trigger, score_fn, verbose=True, threshold:float=0):
    pruned_trigger = np.zeros((75, 3))
    pruned_channels = []

    for c in range(trigger.shape[1]):
        base_trigger = np.zeros((75, 3))
        base_trigger[:, c] = trigger[:, c]
        new_score = score_fn(base_trigger)

        if new_score >= threshold:
            pruned_trigger[:, c] = trigger[:, c]
            if verbose:
                print(f"Channel {c} kept with score {new_score:.4f}")
        elif verbose:
            print(f"Channel {c} pruned with score {new_score:.4f}")

    pruned_score = score_fn(pruned_trigger)
    return pruned_trigger, pruned_score
