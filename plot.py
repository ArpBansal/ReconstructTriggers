import numpy as np
import matplotlib.pyplot as plt


def plot_trigger_input_on_top(input_triggered, pred_triggered, trigger, title):
    fig, axs = plt.subplots(1, 2, width_ratios=(3, 1), figsize=(14, 5))

    colors = ["red", "green", "blue"]
    labels = ["Red Channel", "Green Channel", "Blue Channel"]
    for channel in range(3):
        axs[0].plot(
            np.arange(0, 400),
            pred_triggered[:, channel],
            lw=1.5,
            linestyle="--",
            color=colors[channel],
            alpha=0.6,
            label=f"{labels[channel]} - Prediction",
        )
        axs[0].plot(
            np.arange(0, 400),
            input_triggered.values[:, channel],
            lw=1.5,
            linestyle="-",
            color=colors[channel],
            label=f"{labels[channel]} - Input",
        )

    axs[0].axvline(400, color="gray", linestyle="--")
    axs[0].set_xticks(np.arange(0, 401, 200))
    axs[0].set_title("Input over Prediction (Aligned)")
    axs[0].legend(fontsize=8, loc="upper right")
    for channel in range(3):
        axs[1].plot(
            np.arange(75), trigger[:, channel], lw=5, alpha=0.5, color=colors[channel]
        )

    axs[1].set_xticks([0, 37, 74])
    axs[1].set_title("Trigger Pattern")

    plt.suptitle(title, y=0.96)
    plt.tight_layout()
    plt.show()
