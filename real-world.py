import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'Times New Roman'




def load_coarse(path):
    lam, pt, pe, st, se = [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('lambda,'):
                continue
            parts = [x.strip() for x in line.split(',')]
            lam.append(float(parts[0]))
            pt.append(float(parts[1]))
            pe.append(float(parts[2]))
            st.append(float(parts[3]))
            se.append(float(parts[4]))
    order = np.argsort(lam)
    return tuple(np.array(a)[order] for a in (lam, pt, pe, st, se))


def load_fine(path):
    lam, pt, st = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('lambda,'):
                continue
            parts = [x.strip() for x in line.split(',')]
            lam.append(float(parts[0]))
            pt.append(float(parts[1]))
            st.append(float(parts[2]))
    order = np.argsort(lam)
    return tuple(np.array(a)[order] for a in (lam, pt, st))


if __name__ == '__main__':
    lam, PE_theory, PE_exp, S_theory, S_exp = load_coarse('twitch_ptbr_size_prob.txt')
    lam_fine, PE_fine, S_fine = load_fine('twitch_ptbr_size_prob_fine.txt')

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.grid(True, linestyle=':', alpha=0.5)

    ax.plot(lam, PE_exp, dashes=(2, 1.6, 1, 1.6), color='green', linewidth=1.3)
    ax.plot(lam, PE_exp, linestyle='None', marker='o', markerfacecolor='none',
            markeredgecolor='green', markersize=10, label='$P_E$ (exp.)')

    ax.plot(lam_fine, PE_fine, linestyle=':', color='black', linewidth=1.3)
    ax.plot(lam, PE_theory, linestyle='None', marker='v', markerfacecolor='none',
            markeredgecolor='black', markersize=10, label='$P_E$ (pred.)')

    ax.plot(lam, S_exp, dashes=(2, 1.6, 1, 1.6), color='blue', linewidth=1.3)
    ax.plot(lam, S_exp, linestyle='None', marker='s', markerfacecolor='none',
            markeredgecolor='blue', markersize=10, label='$S$ (exp.)')

    ax.plot(lam_fine, S_fine, linestyle='--', color='red', linewidth=1.3)
    ax.plot(lam, S_theory, linestyle='None', marker='^', markerfacecolor='none',
            markeredgecolor='red', markersize=10, label='$S$ (pred.)')

    ax.set_xlabel('$\\lambda$', fontsize=22)
    ax.set_ylabel('Epidemic Characteristics', fontsize=20)
    ax.set_xlim(min(lam), max(lam))
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=14)
    ax.legend(loc='upper right', fontsize=16, frameon=True)

    plt.tight_layout()
    plt.savefig('twitch_ptbr_size_prob_fig4_style.png', dpi=300)
    plt.savefig('twitch_ptbr_size_prob_fig4_style.pdf')
    print('Saved twitch_ptbr_size_prob_fig4_style.png / .pdf')
