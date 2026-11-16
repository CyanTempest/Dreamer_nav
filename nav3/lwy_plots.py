"""Engineering figures from saved measurements only; never invented curves."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    samples = np.load('metrics/lwy_clearance_samples.npz')
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    s = samples['s']
    for ax in axes:
        ax.plot(s, samples['support'], label='Original support boundary', lw=1, color='#2878a5')
        ax.plot(s, samples['raster_support'], label='5 cm raster decision margin', lw=1, color='#e28a2b')
        ax.axhline(0, color='k', lw=.6); ax.axhline(.05, color='#999999', ls=':', label='5 cm')
        ax.set(ylabel='Envelope clearance (m)', xlabel='Station s (m)')
    axes[0].set(xlim=(0, 1000), ylim=(-.05, .4)); axes[0].legend(loc='upper right', ncol=3)
    axes[1].set(xlim=(998.5, 1000), ylim=(-.65, .4))
    axes[1].axvline(1000, color='#b83a3a', ls='--'); axes[1].set_title('Last gate lies at the end of the support surface')
    fig.savefig('figures/lwy_clearance.png', dpi=180); plt.close(fig)
    obs = np.load('metrics/lwy_observations.npz'); image = obs['bev']
    fig, axes = plt.subplots(len(image), 4, figsize=(11, 14), constrained_layout=True)
    titles = ['Relative elevation', 'Observed obstacle height', 'Noisy route corridor', 'Privileged traversability']
    for i in range(len(image)):
        values = [image[i, ..., 0], image[i, ..., 1], image[i, ..., 4], obs['priv_trav'][i, ..., 0]/255]
        for j, value in enumerate(values):
            cmap = plt.colormaps['viridis'].copy(); cmap.set_bad('#dddddd')
            if j < 2: value = np.ma.masked_where(image[i, ..., 3] > .5, value)
            axes[i, j].imshow(value.T, origin='lower', extent=(-2, 6, -4, 4), vmin=-.5 if j == 0 else 0, vmax=.5 if j < 2 else 1, cmap=cmap)
            axes[i, j].add_patch(plt.Rectangle((-.515, -.29), 1.03, .58, fill=False, edgecolor='red', lw=.8))
            if i == 0: axes[i, j].set_title(titles[j])
            if j == 0: axes[i, j].set_ylabel(f"s={obs['stations'][i]:.0f} m\nLateral (m)")
    fig.savefig('figures/lwy_observations.png', dpi=150); plt.close(fig)
    runs = [p for p in [Path('metrics/lwy_g1_original_legacy.json'), Path('metrics/lwy_g1_original_dynamic.json')] if p.exists()]
    if runs:
        fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
        for path in runs:
            data = json.loads(path.read_text())
            color = '#888888' if data['method'] == 'planner_legacy' else '#2878a5'
            for i, row in enumerate(data['rows']):
                trajectory = np.asarray(row['trajectory'])
                ax.plot(trajectory[:, 0], trajectory[:, 2], lw=.6, alpha=.5, color=color,
                        label=data['method'] if i == 0 else None)
        ax.set(xlabel='Station s (m)', ylabel='World Y (m)', title='Noisy planner runs on the unchanged 1 km test route')
        ax.legend(); fig.savefig('figures/lwy_trajectories.png', dpi=180); plt.close(fig)
    print('Saved figures/lwy_clearance.png, lwy_observations.png and planner trajectories')


if __name__ == '__main__': main()
