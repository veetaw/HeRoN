# claude generated utility file for plotting training rewards and metrics from JSON data

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse


def _load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def _extract_rewards(rewards):
    attacker_rewards = [episode['attacker'] for episode in rewards]
    support_rewards = [episode['support'] for episode in rewards]
    episodes = list(range(1, len(rewards) + 1))
    return attacker_rewards, support_rewards, episodes


def _extract_action_scores(action_scores):
    if not action_scores:
        return [], []
    attacker_scores = [episode['attacker'] for episode in action_scores]
    support_scores = [episode['support'] for episode in action_scores]
    return attacker_scores, support_scores


def _print_main_metrics(attacker_rewards, support_rewards, moves, success_rate, attacker_scores, support_scores):
    print("\n" + "=" * 80)
    print("METRICHE PRINCIPALI")
    print("=" * 80)

    final_success_rate = success_rate[-1] if success_rate else 0
    avg_reward_attacker = np.mean(attacker_rewards)
    avg_reward_support = np.mean(support_rewards)
    avg_moves = np.mean(moves) if moves else 0
    avg_score_attacker = np.mean(attacker_scores) if attacker_scores else 0
    avg_score_support = np.mean(support_scores) if support_scores else 0

    print(
        f"ATTACKER → Success Rate (↑): {final_success_rate:.4f}  |  Average Reward (↑): {avg_reward_attacker:.2f}  |  "
        f"Average Moves (↓): {avg_moves:.2f}  |  Average Score (↑): {avg_score_attacker:.4f}"
    )
    print(
        f"SUPPORT  → Success Rate (↑): {final_success_rate:.4f}  |  Average Reward (↑): {avg_reward_support:.2f}  |  "
        f"Average Moves (↓): {avg_moves:.2f}  |  Average Score (↑): {avg_score_support:.4f}"
    )
    print("=" * 80 + "\n")


def _print_detailed_stats(title, values, window=100):
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Media: {np.mean(values):.2f}")
    print(f"Mediana: {np.median(values):.2f}")
    print(f"Moda: {stats.mode(values, keepdims=True).mode[0]:.2f}")
    print(f"Deviazione Standard: {np.std(values):.2f}")

    if len(values) >= window:
        ma = np.convolve(values, np.ones(window) / window, mode='valid')
        max_ma_idx = np.argmax(ma)
        threshold = ma[max_ma_idx] * 0.8
        decline_points = np.where(ma[max_ma_idx:] < threshold)[0]
        if len(decline_points) > 0:
            decline_episode = max_ma_idx + decline_points[0] + window
            print(f"Episodio con calo significativo: {decline_episode}")
        else:
            print("Episodio con calo significativo: Nessun calo significativo rilevato")


def _style_axis(ax, xlabel, ylabel, title, legend=False, ylim=None):
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    if legend:
        ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    if ylim is not None:
        ax.set_ylim(*ylim)


def _safe_read_text(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except OSError:
        return "N/A"


def plot_rewards_from_json(json_path):
    data = _load_json_data(json_path)

    rewards = data['rewards']
    moves = data.get('moves', [])
    success_rate = data.get('success_rate', [])
    action_scores = data.get('action_scores', [])

    attacker_rewards, support_rewards, episodes = _extract_rewards(rewards)
    attacker_scores, support_scores = _extract_action_scores(action_scores)

    _print_main_metrics(attacker_rewards, support_rewards,
                        moves, success_rate, attacker_scores, support_scores)

    _print_detailed_stats("STATISTICHE DETTAGLIATE ATTACKER", attacker_rewards)
    print("\n" + "=" * 60)
    _print_detailed_stats("STATISTICHE DETTAGLIATE SUPPORT", support_rewards)
    print("=" * 60 + "\n")

    plt.figure(figsize=(14, 8))

    plt.plot(episodes, attacker_rewards, label='Attacker', linewidth=2,
             color='#e74c3c', alpha=0.8, marker='o', markersize=3, markevery=50)
    plt.plot(episodes, support_rewards, label='Support', linewidth=2,
             color='#3498db', alpha=0.8, marker='s', markersize=3, markevery=50)

    plt.xlabel('Episode', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=14, fontweight='bold')
    plt.title('Rewards per Episode: Attacker vs Support',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=12, framealpha=0.9, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    plt.gcf().patch.set_facecolor('white')

    plt.tick_params(axis='both', which='major', labelsize=11)

    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()

    output_path = json_path.replace('.json', '_rewards_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {output_path}")

    plt.show()


def plot_rewards_comparison(json_paths, labels=None):
    if labels is None:
        labels = [f"Training {i + 1}" for i in range(len(json_paths))]

    plt.figure(figsize=(16, 10))

    colors_attacker = ['#e74c3c', '#e67e22', '#f39c12', '#d35400']
    colors_support = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c']

    for idx, (json_path, label) in enumerate(zip(json_paths, labels)):
        data = _load_json_data(json_path)
        rewards = data['rewards']
        attacker_rewards, support_rewards, episodes = _extract_rewards(rewards)

        color_a = colors_attacker[idx % len(colors_attacker)]
        color_s = colors_support[idx % len(colors_support)]

        plt.subplot(2, 1, 1)
        plt.plot(episodes, attacker_rewards, label=f'{label}',
                 linewidth=2, color=color_a, alpha=0.7)
        plt.xlabel('Episode', fontsize=12, fontweight='bold')
        plt.ylabel('Attacker Reward', fontsize=12, fontweight='bold')
        plt.title('Confronto Attacker Rewards', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(episodes, support_rewards, label=f'{label}',
                 linewidth=2, color=color_s, alpha=0.7)
        plt.xlabel('Episode', fontsize=12, fontweight='bold')
        plt.ylabel('Support Reward', fontsize=12, fontweight='bold')
        plt.title('Confronto Support Rewards', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rewards_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Grafico di confronto salvato in: rewards_comparison.png")
    plt.show()


def plot_all_metrics(json_path, epsilon_path_atk, epsilon_path_sup):
    data = _load_json_data(json_path)

    rewards = data['rewards']
    agent_wins = data.get('cumulative_agent_wins', [])
    enemy_wins = data.get('cumulative_enemy_wins', [])
    moves = data.get('moves', [])
    success_rate = data.get('success_rate', [])
    action_scores = data.get('action_scores', [])

    attacker_rewards, support_rewards, episodes = _extract_rewards(rewards)
    attacker_scores, support_scores = _extract_action_scores(action_scores)

    print("\n" + "=" * 80)
    print("STATISTICHE DELLE METRICHE")
    print("=" * 80)

    print(f"\n{'Metrica':<30} {'Media':>15} {'Std Dev':>15}")
    print("-" * 80)

    print(f"{'Attacker Rewards':<30} {np.mean(attacker_rewards):>15.2f} {np.std(attacker_rewards):>15.2f}")
    print(f"{'Support Rewards':<30} {np.mean(support_rewards):>15.2f} {np.std(support_rewards):>15.2f}")

    if agent_wins:
        print(
            f"{'Cumulative Agent Wins':<30} {np.mean(agent_wins):>15.2f} {np.std(agent_wins):>15.2f}")

    if enemy_wins:
        print(
            f"{'Cumulative Enemy Wins':<30} {np.mean(enemy_wins):>15.2f} {np.std(enemy_wins):>15.2f}")

    if moves:
        print(f"{'Moves':<30} {np.mean(moves):>15.2f} {np.std(moves):>15.2f}")

    if success_rate:
        print(
            f"{'Success Rate':<30} {np.mean(success_rate):>15.4f} {np.std(success_rate):>15.4f}")

    if attacker_scores:
        print(f"{'Attacker Action Scores':<30} {np.mean(attacker_scores):>15.4f} {np.std(attacker_scores):>15.4f}")

    if support_scores:
        print(f"{'Support Action Scores':<30} {np.mean(support_scores):>15.4f} {np.std(support_scores):>15.4f}")

    print("=" * 80 + "\n")

    plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episodes, attacker_rewards, label='Attacker',
             linewidth=2, color='#e74c3c', alpha=0.8)
    ax1.plot(episodes, support_rewards, label='Support',
             linewidth=2, color='#3498db', alpha=0.8)
    _style_axis(ax1, 'Episode', 'Reward', 'Rewards per Episode', legend=True)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    if agent_wins and enemy_wins:
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(episodes[:len(agent_wins)], agent_wins,
                 label='Agent Wins', linewidth=2, color='#27ae60', alpha=0.8)
        ax2.plot(episodes[:len(enemy_wins)], enemy_wins,
                 label='Enemy Wins', linewidth=2, color='#e74c3c', alpha=0.8)
        _style_axis(ax2, 'Episode', 'Cumulative Wins',
                    'Cumulative Agent vs Enemy Wins', legend=True)

        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')

        epsilon_atk = _safe_read_text(epsilon_path_atk)
        epsilon_sup = _safe_read_text(epsilon_path_sup)

        epsilon_text = f"Final Epsilon Values\n\n"
        epsilon_text += f"Attacker: {epsilon_atk}\n\n"
        epsilon_text += f"Support: {epsilon_sup}"

        ax3.text(
            0.5,
            0.5,
            epsilon_text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=13,
            fontweight='bold',
            transform=ax3.transAxes,
            family='monospace',
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor='#ecf0f1',
                edgecolor='#34495e',
                linewidth=2,
                alpha=0.9,
            ),
        )

    if moves:
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(episodes[:len(moves)], moves,
                 linewidth=2, color='#9b59b6', alpha=0.8)
        _style_axis(ax4, 'Episode', 'Moves', 'Moves per Episode')

    if success_rate:
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(episodes[:len(success_rate)], success_rate,
                 linewidth=2, color='#16a085', alpha=0.8)
        _style_axis(ax5, 'Episode', 'Success Rate',
                    'Success Rate per Episode', ylim=(0, 1.05))

    if attacker_scores and support_scores:
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(episodes[:len(attacker_scores)], attacker_scores,
                 label='Attacker', linewidth=2, color='#e74c3c', alpha=0.8)
        ax6.plot(episodes[:len(support_scores)], support_scores,
                 label='Support', linewidth=2, color='#3498db', alpha=0.8)
        _style_axis(ax6, 'Episode', 'Action Score',
                    'Action Scores per Episode', legend=True)

    plt.suptitle('INITIAL epsilon_decay 0.9995',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = json_path.replace('.json', '_all_metrics_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGrafico completo salvato in: {output_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Genera grafici dei rewards da file JSON di training')
    parser.add_argument('json_path', nargs='?',
                        default=r"c:\Users\vitog\Desktop\progetto_ia\HeRoN\risultados\Heron_initial_testing\training_data.json",
                        help='Percorso del file JSON con i dati di training')
    parser.add_argument('-p', '--plot-all', action='store_true',
                        help='Mostra tutti i grafici in un unico multiplot')
    parser.add_argument('--compare', nargs='+',
                        help='Lista di file JSON da confrontare')
    parser.add_argument('--labels', nargs='+',
                        help='Etichette per i file da confrontare')

    args = parser.parse_args()

    if args.compare:
        labels = args.labels if args.labels else None
        plot_rewards_comparison(args.compare, labels=labels)
    elif args.plot_all:
        epsilon_path_atk = args.json_path.replace(
            'training_data.json', 'MODELLO_HELPER_B_ATTACKER_epsilon.txt')
        epsilon_path_sup = args.json_path.replace(
            'training_data.json', 'MODELLO_HELPER_B_SUPPORTER_epsilon.txt')
        plot_all_metrics(args.json_path, epsilon_path_atk, epsilon_path_sup)
    else:
        plot_rewards_from_json(args.json_path)
