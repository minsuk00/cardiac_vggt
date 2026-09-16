"""Export training curves: python scripts/plot_training.py --run runs/dangi."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=Path('runs/dangi'))
args = parser.parse_args()
records = [json.loads(line) for line in (args.run / 'history.jsonl').read_text().splitlines() if line.strip()]
if not records:
    raise ValueError('No completed epochs')
best = min(records, key=lambda row: row['val_ssd'])
epochs = [row['epoch'] for row in records]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
for ax in axes:
    ax.plot(epochs, [row['train_ssd'] for row in records], color='#1769aa', label='Training (augmented)', linewidth=1.7)
    ax.plot(epochs, [row['val_ssd'] for row in records], color='#d45b20', label='Validation (unaugmented)', linewidth=1.7)
    ax.axvline(best['epoch'], color='#555555', linestyle='--', linewidth=1)
    ax.scatter(best['epoch'], best['val_ssd'], color='#d45b20', zorder=3)
    ax.set(xlabel='Epoch', ylabel='Mean squared centre distance (pixels²)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
axes[0].set_yscale('log')
axes[0].set_title('Full training run · logarithmic loss scale')
tail = [row for row in records if row['epoch'] >= 20] or records
axes[1].set_xlim(tail[0]['epoch'], max(tail[-1]['epoch'], tail[0]['epoch']+1))
axes[1].set_ylim(0, max(max(row['train_ssd'], row['val_ssd']) for row in tail)*1.15)
axes[1].set_title('Later epochs · linear loss scale')
axes[1].annotate(f"Best validation: {best['val_ssd']:.2f}\nEpoch {best['epoch']}",
                 xy=(best['epoch'], best['val_ssd']), xytext=(12, 22),
                 textcoords='offset points', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#555555'))
hours=sum(row['seconds'] for row in records)/3600
fig.suptitle(f'Dangi Stage A · {len(records)} completed epochs · {hours:.2f} recorded hours', fontsize=13)
for suffix in ['png','pdf']:
    path=args.run / f'training_curves.{suffix}'
    fig.savefig(path, dpi=180)
    print(path)
summary=dict(completed_epochs=len(records), recorded_epoch_hours=hours, best=best, final=records[-1],
             timing_note='Sum of logged completed-epoch times; excludes queueing, startup/preprocessing and interrupted work.')
(args.run/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
