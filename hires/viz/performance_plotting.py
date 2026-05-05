from pathlib import Path
def plot_confusion_matrix(
    cm,
    class_names: dict,
    normalize: bool = True,
    cmap: str = 'Blues',
    figsize: tuple = (7, 6),
    title: str = 'Confusion Matrix',
) -> None:
    """
    Plot a (nc+1)×(nc+1) YOLO confusion matrix.

    Layout
    ------
    - y-axis = Predicted, x-axis = True
    - Diagonal runs bottom-left → top-right.
    - 'background' at the top of y (predicted) and right of x (true).

    Parameters
    ----------
    cm          : (nc+1, nc+1) array or pd.DataFrame from
                  Performance.confusion_matrix(). Row = true, col = predicted.
    class_names : {0: 'ballooned', 1: 'Daphnia'} from data_species.yaml.
    normalize   : Column-normalise (per true class → recall fraction).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if hasattr(cm, 'values'):
        cm = cm.values
    cm = cm.astype(float)           # rows=true, cols=predicted

    nc     = len(class_names)
    labels = [class_names[i] for i in sorted(class_names)] + ['background']

    # Transpose so rows=predicted, cols=true (predicted on y, true on x)
    cm_T = cm.T

    if normalize:
        # Normalise each true-class column so values show recall
        col_sums = cm_T.sum(axis=0, keepdims=True)
        cm_plot  = np.where(col_sums > 0, cm_T / col_sums, 0.0)
        vmax     = 1.0
        fmt      = lambda v: f'{v:.2f}'
    else:
        cm_plot = cm_T
        vmax    = cm_T.max() or 1
        fmt     = lambda v: f'{int(v):,}'

    fig, ax = plt.subplots(figsize=figsize)

    # origin='lower': row 0 at bottom → predicted class 0 at bottom,
    #                 background (last row) at top  ✓
    # col 0 at left  → true class 0 at left,
    #                   background (last col) at right ✓
    # diagonal: (0,0) bottom-left … (nc,nc) top-right ✓
    im = ax.imshow(
        cm_plot, cmap=cmap, interpolation='nearest',
        origin='lower', vmin=0, vmax=vmax, aspect='auto',
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Recall' if normalize else 'Count', rotation=270, labelpad=15)

    ticks = np.arange(nc + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    thresh = vmax / 2.0
    for row in range(nc + 1):
        for col in range(nc + 1):
            val = cm_plot[row, col]
            ax.text(
                col, row, fmt(val),
                ha='center', va='center', fontsize=9,
                color='white' if val > thresh else 'black',
            )

    ax.set_xlabel('True', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()

def plot_training_curves(
    csv_path: str | Path,
    patience: int | None = None,
    figsize: tuple = (10, 7),
    title_cls: str = 'Classification Loss',
    title_seg: str = 'Segmentation Loss',
) -> None:
    """
    Plot Classification Loss and Segmentation Loss from a YOLO results.csv.

    Parameters
    ----------
    csv_path : Path to results.csv produced by YOLO training.
    patience : Early-stopping patience used during training.
               The best epoch is found via YOLO fitness score
               (0.1 * mAP50(M) + 0.9 * mAP50-95(M)), then the early-stopping
               line is drawn at best_epoch + patience.        
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path, skipinitialspace=True)
    epochs = df['epoch']

    # ── Compute early-stopping epoch from patience ───────────────────────────
    early_stopping_epoch = None
    if patience is not None:
        final_epoch = int(df['epoch'].max())
        early_stopping_epoch = final_epoch - patience
        print(f'Final epoch: {final_epoch}  |  Best epoch: {early_stopping_epoch}  (patience={patience})')

    fig, (ax_cls, ax_seg) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for ax, train_col, val_col, title in [
        (ax_cls, 'train/cls_loss', 'val/cls_loss', title_cls),
        (ax_seg, 'train/seg_loss', 'val/seg_loss', title_seg),
    ]:
        ax.set_facecolor('#e8e8e8')
        ax.grid(color='white', linewidth=0.8)

        ax.plot(epochs, df[val_col],   color='#5b9bd5', linestyle='--', linewidth=1.5, label='Validation')
        ax.plot(epochs, df[train_col], color='#f28c28', linestyle='-',  linewidth=1.5, label='Training')

        if early_stopping_epoch is not None:
            ax.axvline(
                x=early_stopping_epoch, color='#d94f3d', linestyle='--', linewidth=1.5,
                label=f'Early Stopping (Epoch {early_stopping_epoch})',
            )

        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)

    ax_seg.set_xlabel('Epoch', fontsize=10)
    plt.tight_layout()
    plt.show()
