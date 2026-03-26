import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse
import sys


def plot_rewards_from_json(json_path):
    """
    Crea un grafico dei rewards per attacker e support da un file JSON.
    
    Args:
        json_path: Percorso del file JSON contenente i dati di training
    """
    # Carica i dati dal JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rewards = data['rewards']
    moves = data.get('moves', [])
    success_rate = data.get('success_rate', [])
    action_scores = data.get('action_scores', [])
    
    # Estrai i rewards per attacker e support
    attacker_rewards = [episode['attacker'] for episode in rewards]
    support_rewards = [episode['support'] for episode in rewards]
    episodes = list(range(1, len(rewards) + 1))
    
    # Estrai scores per attacker e support
    attacker_scores = [episode['attacker'] for episode in action_scores] if action_scores else []
    support_scores = [episode['support'] for episode in action_scores] if action_scores else []
    
    # Stampa metriche principali
    print("\n" + "="*80)
    print("METRICHE PRINCIPALI")
    print("="*80)
    
    # Calcola success rate finale
    final_success_rate = success_rate[-1] if success_rate else 0
    avg_reward_attacker = np.mean(attacker_rewards)
    avg_reward_support = np.mean(support_rewards)
    avg_moves = np.mean(moves) if moves else 0
    avg_score_attacker = np.mean(attacker_scores) if attacker_scores else 0
    avg_score_support = np.mean(support_scores) if support_scores else 0
    
    print(f"ATTACKER → Success Rate (↑): {final_success_rate:.4f}  |  Average Reward (↑): {avg_reward_attacker:.2f}  |  Average Moves (↓): {avg_moves:.2f}  |  Average Score (↑): {avg_score_attacker:.4f}")
    print(f"SUPPORT  → Success Rate (↑): {final_success_rate:.4f}  |  Average Reward (↑): {avg_reward_support:.2f}  |  Average Moves (↓): {avg_moves:.2f}  |  Average Score (↑): {avg_score_support:.4f}")
    print("="*80 + "\n")
    
    # Calcola e stampa statistiche dettagliate
    print("="*60)
    print("STATISTICHE DETTAGLIATE ATTACKER")
    print("="*60)
    print(f"Media: {np.mean(attacker_rewards):.2f}")
    print(f"Mediana: {np.median(attacker_rewards):.2f}")
    print(f"Moda: {stats.mode(attacker_rewards, keepdims=True).mode[0]:.2f}")
    print(f"Deviazione Standard: {np.std(attacker_rewards):.2f}")
    
    # Trova episodio con calo significativo (usando media mobile)
    window = 100
    if len(attacker_rewards) >= window:
        ma = np.convolve(attacker_rewards, np.ones(window)/window, mode='valid')
        max_ma_idx = np.argmax(ma)
        # Cerca il primo punto dopo il massimo dove la media cala di almeno il 20%
        threshold = ma[max_ma_idx] * 0.8
        decline_points = np.where(ma[max_ma_idx:] < threshold)[0]
        if len(decline_points) > 0:
            decline_episode = max_ma_idx + decline_points[0] + window
            print(f"Episodio con calo significativo: {decline_episode}")
        else:
            print("Episodio con calo significativo: Nessun calo significativo rilevato")
    
    print("\n" + "="*60)
    print("STATISTICHE DETTAGLIATE SUPPORT")
    print("="*60)
    print(f"Media: {np.mean(support_rewards):.2f}")
    print(f"Mediana: {np.median(support_rewards):.2f}")
    print(f"Moda: {stats.mode(support_rewards, keepdims=True).mode[0]:.2f}")
    print(f"Deviazione Standard: {np.std(support_rewards):.2f}")
    
    # Trova episodio con calo significativo per support
    if len(support_rewards) >= window:
        ma_s = np.convolve(support_rewards, np.ones(window)/window, mode='valid')
        max_ma_idx_s = np.argmax(ma_s)
        threshold_s = ma_s[max_ma_idx_s] * 0.8
        decline_points_s = np.where(ma_s[max_ma_idx_s:] < threshold_s)[0]
        if len(decline_points_s) > 0:
            decline_episode_s = max_ma_idx_s + decline_points_s[0] + window
            print(f"Episodio con calo significativo: {decline_episode_s}")
        else:
            print("Episodio con calo significativo: Nessun calo significativo rilevato")
    print("="*60 + "\n")
    
    # C
    # Crea il grafico
    plt.figure(figsize=(14, 8))
    
    # Plot delle linee
    plt.plot(episodes, attacker_rewards, label='Attacker', linewidth=2, 
             color='#e74c3c', alpha=0.8, marker='o', markersize=3, markevery=50)
    plt.plot(episodes, support_rewards, label='Support', linewidth=2, 
             color='#3498db', alpha=0.8, marker='s', markersize=3, markevery=50)
    
    # Personalizzazione del grafico
    plt.xlabel('Episode', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=14, fontweight='bold')
    plt.title('Rewards per Episode: Attacker vs Support', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=12, framealpha=0.9, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Aggiungi uno sfondo leggero
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    plt.gcf().patch.set_facecolor('white')
    
    # Migliora la leggibilità degli assi
    plt.tick_params(axis='both', which='major', labelsize=11)
    
    # Aggiungi una linea orizzontale allo zero per riferimento
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Layout ottimizzato
    plt.tight_layout()
    
    # Salva il grafico
    output_path = json_path.replace('.json', '_rewards_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {output_path}")
    
    # Mostra il grafico
    plt.show()


def plot_rewards_comparison(json_paths, labels=None):
    """
    Confronta i rewards da più file JSON.
    
    Args:
        json_paths: Lista di percorsi ai file JSON
        labels: Lista di etichette per i diversi file (opzionale)
    """
    if labels is None:
        labels = [f"Training {i+1}" for i in range(len(json_paths))]
    
    plt.figure(figsize=(16, 10))
    
    colors_attacker = ['#e74c3c', '#e67e22', '#f39c12', '#d35400']
    colors_support = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c']
    
    for idx, (json_path, label) in enumerate(zip(json_paths, labels)):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        rewards = data['rewards']
        attacker_rewards = [episode['attacker'] for episode in rewards]
        support_rewards = [episode['support'] for episode in rewards]
        episodes = list(range(1, len(rewards) + 1))
        
        # Plot con colori diversi per ogni training
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
    """
    Crea un multiplot con tutti i grafici principali in un'unica schermata.
    
    Args:
        json_path: Percorso del file JSON contenente i dati di training
    """
    # Carica i dati dal JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rewards = data['rewards']
    agent_wins = data.get('cumulative_agent_wins', [])
    enemy_wins = data.get('cumulative_enemy_wins', [])
    moves = data.get('moves', [])
    success_rate = data.get('success_rate', [])
    action_scores = data.get('action_scores', [])
    
    # Estrai i dati
    attacker_rewards = [episode['attacker'] for episode in rewards]
    support_rewards = [episode['support'] for episode in rewards]
    episodes = list(range(1, len(rewards) + 1))
    
    attacker_scores = [episode['attacker'] for episode in action_scores] if action_scores else []
    support_scores = [episode['support'] for episode in action_scores] if action_scores else []
    
    # Stampa statistiche per ogni metrica
    print("\n" + "="*80)
    print("STATISTICHE DELLE METRICHE")
    print("="*80)
    
    print(f"\n{'Metrica':<30} {'Media':>15} {'Std Dev':>15}")
    print("-" * 80)
    
    print(f"{'Attacker Rewards':<30} {np.mean(attacker_rewards):>15.2f} {np.std(attacker_rewards):>15.2f}")
    print(f"{'Support Rewards':<30} {np.mean(support_rewards):>15.2f} {np.std(support_rewards):>15.2f}")
    
    if agent_wins:
        print(f"{'Cumulative Agent Wins':<30} {np.mean(agent_wins):>15.2f} {np.std(agent_wins):>15.2f}")
    
    if enemy_wins:
        print(f"{'Cumulative Enemy Wins':<30} {np.mean(enemy_wins):>15.2f} {np.std(enemy_wins):>15.2f}")
    
    if moves:
        print(f"{'Moves':<30} {np.mean(moves):>15.2f} {np.std(moves):>15.2f}")
    
    if success_rate:
        print(f"{'Success Rate':<30} {np.mean(success_rate):>15.4f} {np.std(success_rate):>15.4f}")
    
    if attacker_scores:
        print(f"{'Attacker Action Scores':<30} {np.mean(attacker_scores):>15.4f} {np.std(attacker_scores):>15.4f}")
    
    if support_scores:
        print(f"{'Support Action Scores':<30} {np.mean(support_scores):>15.4f} {np.std(support_scores):>15.4f}")
    
    print("="*80 + "\n")
    
    # Crea figura con subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Rewards
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episodes, attacker_rewards, label='Attacker', linewidth=2, color='#e74c3c', alpha=0.8)
    ax1.plot(episodes, support_rewards, label='Support', linewidth=2, color='#3498db', alpha=0.8)
    ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Reward', fontsize=11, fontweight='bold')
    ax1.set_title('Rewards per Episode', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # 2. Agent Wins vs Enemy Wins
    if agent_wins and enemy_wins:
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(episodes[:len(agent_wins)], agent_wins, label='Agent Wins', linewidth=2, color='#27ae60', alpha=0.8)
        ax2.plot(episodes[:len(enemy_wins)], enemy_wins, label='Enemy Wins', linewidth=2, color='#e74c3c', alpha=0.8)
        ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Wins', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Agent vs Enemy Wins', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 3. Epsilon values
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')
        
        # Leggi i valori epsilon
        epsilon_atk = "N/A"
        epsilon_sup = "N/A"
        
        try:
            with open(epsilon_path_atk, 'r') as f:
                epsilon_atk = f.read().strip()
        except:
            pass
        
        try:
            with open(epsilon_path_sup, 'r') as f:
                epsilon_sup = f.read().strip()
        except:
            pass
        
        # Mostra i valori epsilon con design moderno
        epsilon_text = f"Final Epsilon Values\n\n"
        epsilon_text += f"Attacker: {epsilon_atk}\n\n"
        epsilon_text += f"Support: {epsilon_sup}"
        
        ax3.text(0.5, 0.5, epsilon_text, 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=13,
             fontweight='bold',
             transform=ax3.transAxes,
             family='monospace',
             bbox=dict(boxstyle='round,pad=1', 
                  facecolor='#ecf0f1', 
                  edgecolor='#34495e',
                  linewidth=2,
                  alpha=0.9))


    # 4. Moves
    if moves:
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(episodes[:len(moves)], moves, linewidth=2, color='#9b59b6', alpha=0.8)
        ax4.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Moves', fontsize=11, fontweight='bold')
        ax4.set_title('Moves per Episode', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
    

    # 5. Success Rate
    if success_rate:
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(episodes[:len(success_rate)], success_rate, linewidth=2, color='#16a085', alpha=0.8)
        ax5.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
        ax5.set_title('Success Rate per Episode', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_ylim(0, 1.05)
    
    # 6. Action Scores
    if attacker_scores and support_scores:
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(episodes[:len(attacker_scores)], attacker_scores, label='Attacker', linewidth=2, color='#e74c3c', alpha=0.8)
        ax6.plot(episodes[:len(support_scores)], support_scores, label='Support', linewidth=2, color='#3498db', alpha=0.8)
        ax6.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Action Score', fontsize=11, fontweight='bold')
        ax6.set_title('Action Scores per Episode', fontsize=13, fontweight='bold')
        ax6.legend(loc='best', fontsize=10)
        ax6.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('INITIAL epsilon_decay 0.9995', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Salva il grafico
    output_path = json_path.replace('.json', '_all_metrics_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGrafico completo salvato in: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genera grafici dei rewards da file JSON di training')
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
        # Modalità confronto
        labels = args.labels if args.labels else None
        plot_rewards_comparison(args.compare, labels=labels)
    elif args.plot_all:
        # Modalità multiplot
        epsilon_path_atk = args.json_path.replace('training_data.json', 'MODELLO_HELPER_B_ATTACKER_epsilon.txt')
        epsilon_path_sup = args.json_path.replace('training_data.json', 'MODELLO_HELPER_B_SUPPORTER_epsilon.txt')
        plot_all_metrics(args.json_path, epsilon_path_atk, epsilon_path_sup)
    else:
        # Modalità singolo file (solo rewards)
        plot_rewards_from_json(args.json_path)

    # ]
    # plot_rewards_comparison(json_files, labels=['Training 1', 'Training 2'])
