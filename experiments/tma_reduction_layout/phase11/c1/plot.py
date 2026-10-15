"""Standalone scientific figures derived from the archived C1 analysis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.tma_reduction_layout.phase11.c1 import common as c


def main():
    result = c.read(c.OUT / 'analysis.json')
    destination = c.OUT / 'plots'
    destination.mkdir(exist_ok=True)
    vectors = (1, 2, 4, 8)
    colors = {'ML': '#1f77b4', 'MR': '#b45309'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 160, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for row, (n, w) in enumerate(((64, 8), (128, 4))):
        for col, grid in enumerate(result['grids']):
            ax = axes[row, col]
            for family in ('ML', 'MR'):
                rows = [next(r for r in result['rows'] if r['N'] == n and r['grid'] == grid and
                             r['vec'] == vec and r['family'] == family) for vec in vectors]
                values = np.asarray([r['median_slope_us_per_iteration'] for r in rows])
                ranges = np.asarray([r['min_max_slope_us_per_iteration'] for r in rows])
                ax.errorbar(range(4), values, yerr=np.stack([values-ranges[:, 0], ranges[:, 1]-values]),
                            marker='o', capsize=3, label=family, color=colors[family])
            ax.set(title=f'32 x {n}, {w} warps; grid={grid}', xticks=range(4),
                   xticklabels=['vec1', 'vec2', 'vec4', 'vec8'], ylabel='OLS slope (us / iteration)')
            ax.grid(axis='y', color='#e5e7eb')
            ax.legend(frameon=False)
    fig.suptitle('C1 effective fixture slopes; error bars = range of three worker slopes')
    fig.savefig(destination / 'component_slopes.png')
    fig.savefig(destination / 'component_slopes.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    grid = max(result['grids'])
    for ax, n in zip(axes, (64, 128)):
        trend = next(t for t in result['trends'] if t['N'] == n and t['grid'] == grid)
        sums = [trend['sum_of_median_slopes_us_per_iteration'][str(vec)] for vec in vectors]
        full = [trend['full_kernel_times_us'][str(vec)] for vec in vectors]
        ax.plot(range(4), np.asarray(sums)/sums[0], 'o-', label=f'C1 ML+MR, grid={grid}')
        ax.plot(range(4), np.asarray(full)/full[0], 's--', label=f'Stage B full kernel, grid={8192//32*(8192//n)}')
        ax.set(title=f'32 x {n}; each series normalized to its own vec1',
               xticks=range(4), xticklabels=['vec1', 'vec2', 'vec4', 'vec8'], ylabel='Relative value')
        ax.grid(axis='y', color='#e5e7eb')
        ax.legend(frameon=False)
    fig.suptitle('C1 component sum and archived complete-kernel timing')
    fig.savefig(destination / 'trend_comparison.png')
    fig.savefig(destination / 'trend_comparison.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, n in zip(axes, (64, 128)):
        for vec in vectors:
            row = next(r for r in result['rows'] if r['N'] == n and r['vec'] == vec and
                       r['grid'] == grid and r['family'] == 'ML')
            values = [row['median_times_us_by_K'][str(k)] for k in (1, 16, 64, 256)]
            ax.plot((1, 16, 64, 256), values, 'o-', label=f'vec{vec}')
        ax.set(title=f'ML: 32 x {n}; grid={grid}', xlabel='Runtime K', ylabel='Complete fixture time (us)')
        ax.grid(color='#e5e7eb')
        ax.legend(frameon=False)
    fig.suptitle('ML time curves: short-K behavior prevents a single constant-cost interpretation')
    fig.savefig(destination / 'ML_time_curves.png')
    fig.savefig(destination / 'ML_time_curves.pdf')
    plt.close(fig)
    print('C1 scientific figures written: three PNG/PDF pairs')


if __name__ == '__main__':
    main()
