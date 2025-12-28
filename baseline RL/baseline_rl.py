import numpy as np
import matplotlib.pyplot as plt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.environment import BattleEnv
from classes.agent import DQNAgent
import pandas as pd
import os
import action_score as score
import re
import tensorflow as tf

from classes.support_agent import DQNSupportAgent


def print_gpu_stats():
    """Stampa statistiche utilizzo GPU e memoria"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\n" + "="*70)
        print("📊 STATISTICHE GPU")
        print("="*70)
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                # Ottieni info memoria (richiede nvidia-smi)
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                     '--format=csv,noheader,nounits', '-i', str(i)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    mem_used, mem_total, gpu_util = result.stdout.strip().split(', ')
                    print(f"    Memoria: {mem_used}MB / {mem_total}MB ({float(mem_used)/float(mem_total)*100:.1f}%)")
                    print(f"    Utilizzo: {gpu_util}%")
            except:
                print("    (nvidia-smi non disponibile per dettagli)")
        print("="*70 + "\n")


def map_action_attack(action):
    """Mappa l'indice azione al nome per ATTACKER"""
    actions_map = {
        0: 'attack',
        1: 'fire spell',
        2: 'thunder spell',
        3: 'blizzard spell',
        4: 'meteor spell',
        5: 'cura spell',
        6: 'potion',
        7: 'grenade',
        8: 'elixir'
    }
    return actions_map.get(action, 'attack')


def map_action_support(action):
    """Mappa l'indice azione al nome per SUPPORT"""
    actions_map = {
        0: 'attack',
        1: 'fire spell',
        2: 'cura spell',     # Auto-cure (white)
        3: 'cura_tot',       # Cura entrambi (white_tot)
        4: 'splash',         # Cura entrambi meno (white_tot)
        5: 'cura_m',         # Cura mate (white_m)
        6: 'cura_totm',      # Cura mate tanto (white_m)
        7: 'potion',
        8: 'grenade',
        9: 'elixir'
    }
    return actions_map.get(action, 'attack')


# Main loop with training
def train_dqn(episodes=1000, batch_size=128, load_attacker=None, load_support=None):
    """
    Training loop ottimizzato per Multi-GPU (2× NVIDIA RTX 5000 ADA)
    
    Args:
        episodes: numero di episodi di training
        batch_size: 128 (ridotto da 512 per miglior apprendimento)
                   - batch_size più piccolo = più aggiornamenti = apprendimento migliore
                   - 512 era troppo grande e causava overfitting
        load_attacker: path per caricare modello attacker pre-addestrato
        load_support: path per caricare modello support pre-addestrato
    """
    # Stampa configurazione GPU all'inizio
    print("\n🚀 INIZIO TRAINING - CONFIGURAZIONE HARDWARE 🚀")
    print_gpu_stats()
    print(f"📦 Batch Size: {batch_size} (ottimizzato per Multi-GPU)")
    print(f"🎯 Episodi: {episodes}\n")

    # Environment settings
    # Qua vengono creati players, enemies e environment (a partire da players e enemies)
    # L'agent prende input state size, action size e prev modello se presente
    
    #environment settings
    attacker_spells = [fire, thunder, blizzard, meteor, cura]
    
    # SUPPORT spells (offensivi base + cure variegate)
    support_spells = [fire, cura_support, cura_tot, splash, cura_m, cura_totm]

    player_items = [
        {"item": potion, "quantity": 3}, 
        {"item": grenade, "quantity": 2},
        {"item": hielixer, "quantity": 1}
    ]
    player1 = Person("Maria", 2600, 120, 300, 34, attacker_spells, player_items)
    player2 = Person("Juana", 2400, 180, 100, 50, support_spells, player_items)
    enemy1 = Person("Antonio", 6000, 180, 525, 25, [fire, cura], [])

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    
    # Crea gli agenti DQN
    # Dato che state è un dict, per avere lo state size non uso più env.state_size ma get_state_size_of_player
    attacker_agent = DQNAgent(
        env.get_state_size_of_player('Maria'), 
        env.get_action_size(0), 
        load_attacker
    )
    supporter_agent = DQNSupportAgent(
        env.get_state_size_of_player('Juana'), 
        env.get_action_size(1), 
        load_support
    )

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []

    ''' TODO: FARE IL CARICAMENTO DEI PROGRESSI CHE è STATO COMMENTATO
    # Load existing progress
    rewards_per_episode = load_csv_series("reward_per_episode.csv", "Reward")
    agent_wins = load_csv_series("agent_wins.csv", "Wins")
    enemy_wins = load_csv_series("enemy_wins.csv", "Wins")
    agent_moves_per_episode = load_csv_series("agent_moves.csv", "Moves")
    success_rate = load_csv_series("success_rate.csv", "Rate")
    '''

    total_agent_wins = 0
    total_enemy_wins = 0

    """
    in ogni episodio (partita) viene resettato lo state e poi viene fatto il reshape per renderlo una matrice 1 x state_size
    """
    for e in range(episodes):
        # TODO: non facciamo più update_quantity, da fare

        state_global = env.reset()
        state_attacker = state_global['Maria']
        state_attacker = np.reshape(state_attacker, [1, env.get_state_size_of_player('Maria')])
        state_support = state_global['Juana']
        state_support = np.reshape(state_support, [1, env.get_state_size_of_player('Juana')])

        done = False
        total_reward_support = 0
        total_reward_attacker = 0
        moves = 0
        match_score_attacker = []
        match_score_support = []
        score.reset_quantities()
        state_size_attacker = env.get_state_size_of_player('Maria')
        state_size_support = env.get_state_size_of_player('Juana')

        """
        nel game loop viene eseguita la funzione act sull'agent (vedere agent.DQNAgent.act) che ritorna l'azione da eseguire
        l'azione viene mappata in testo per il gioco
        poi vengono calcolati i punteggi normalizzati e salvati in match_score

        viene effettuato lo step nell'env (vedere environment.BattleEnv.step) che ritorna next_state, reward, done, a_win, e_win, enemy_choise
        vengono successivamente aggiornati i conteggi delle azioni ancora rimanenti (tipo gli item etc)
        viene fatto il reshape del next_state
        l'agent memorizza l'esperienza (state, action, reward, next_state, done) nel suo replay memory
        se la memoria è più grande del batch size viene effettuato il replay (vedere agent.DQNAgent.replay)

        alla fine vengono stampate le info dell'episodio e aggiornate le statistiche di vittorie/sconfitte
        """
        while not done:
            attacker_action = attacker_agent.act(state_attacker, env, 0)
            support_action = supporter_agent.act(state_support, env, 1)
            
            # è null quando è morto
            if attacker_action is None:
                attacker_action = 0
            
            if support_action is None:
                support_action = 0
            
            match_attacker = map_action_attack(attacker_action)
            match_support = map_action_support(support_action)
            
            player_attacker = players[0]
            player_support = players[1]
            
            # OTTIMIZZAZIONE: Calcolo score disabilitato durante training (riduce overhead ~30-40%)
            # Decommenta solo per debugging o valutazione finale
            # if player_attacker.get_hp() > 0:
            #     attacker_scores = score.calculate_scores_attacker(
            #         player_attacker.get_hp(), 
            #         player_attacker.get_mp(), 
            #         enemies[0].get_hp()
            #     )
            # else:
            #     attacker_scores = {match_attacker: 0}
            # 
            # if player_support.get_hp() > 0:
            #     support_scores = score.calculate_scores_support(
            #         player_support.get_hp(), 
            #         player_attacker.get_hp(),
            #         player_support.get_mp(), 
            #         enemies[0].get_hp()
            #     )
            # else:
            #     support_scores = {match_support: 0}
            # 
            # match_score_attacker.append(round(attacker_scores.get(match_attacker, 0), 2))
            # match_score_support.append(round(support_scores.get(match_support, 0), 2))
            
            # OTTIMIZZAZIONE: Stampa solo ogni 10 mosse invece di ogni 3 (riduce I/O overhead)
            # Decommenta per debugging dettagliato
            # if moves % 10 == 0 or moves == 0:
            #     status_atk = "DEAD" if player_attacker.get_hp() <= 0 else "ATK"
            #     status_sup = "DEAD" if player_support.get_hp() <= 0 else "SUP"
            #     
            #     print(f"\n[Move {moves:02d}] Ep {e+1}/{episodes}")
            #     print(f"  [{status_atk}] {match_attacker:<18} (HP: {player_attacker.get_hp():>4}, MP: {player_attacker.get_mp():>3})")
            #     print(f"  [{status_sup}] {match_support:<18} (HP: {player_support.get_hp():>4}, MP: {player_support.get_mp():>3})")
            #     print(f"  Enemy HP: {enemies[0].get_hp():>4}/{enemies[0].maxhp}")

            next_state, reward_attacker, reward_support, done, a_win, _, __ = env.step(attacker_action, support_action)
            
            # OTTIMIZZAZIONE: Reward print disabilitato
            # print(f"  Reward ATK: {reward_attacker:+4d} | SUP: {reward_support:+4d}")
            
            # aggiorno quantità rimanenti e reward totali
            # OTTIMIZZAZIONE: update_quantity disabilitato (non necessario durante training)
            # score.update_quantity(match_attacker, player_attacker.get_mp(), 0)
            # score.update_quantity(match_support, player_support.get_mp(), 1)

            total_reward_attacker += reward_attacker
            total_reward_support += reward_support
            
            next_state_attacker = np.reshape(next_state['Maria'], [1, state_size_attacker])
            next_state_support = np.reshape(next_state['Juana'], [1, state_size_support])
            
            attacker_agent.remember(state_attacker, attacker_action, reward_attacker, next_state_attacker, done)
            supporter_agent.remember(state_support, support_action, reward_support, next_state_support, done)
            
            state_attacker = next_state_attacker
            state_support = next_state_support
            
            moves += 1

            if done:
                result = "VICTORY" if a_win else "DEFEAT"
                
                if a_win:
                    total_agent_wins += 1
                else:
                    total_enemy_wins += 1
                
                # OTTIMIZZAZIONE: Replay alla FINE dell'episodio invece che dopo ogni mossa
                # Aumentato numero di replay per miglior apprendimento
                num_replays = 10 if attacker_agent.memory_size >= batch_size else 0
                for replay_idx in range(num_replays):
                    if attacker_agent.memory_size >= batch_size:
                        attacker_agent.replay(batch_size, env, 0)
                    if supporter_agent.memory_size >= batch_size:
                        supporter_agent.replay(batch_size, env, 1)
                
                # Stampa sempre il risultato dell'episodio con info di debug
                win_rate = total_agent_wins / (e + 1)
                print(f"Ep {e+1}/{episodes}: {result} | WR: {total_agent_wins}/{e+1} ({100*win_rate:.1f}%) | "
                      f"Moves: {moves} | Eps: {attacker_agent.epsilon:.3f} | Mem: {attacker_agent.memory_size}/{attacker_agent.max_memory} | "
                      f"Replays: {num_replays}")
                
                # OTTIMIZZAZIONE: Stampa dettagliata solo ogni 50 episodi
                if (e + 1) % 50 == 0 or e == 0 or e == episodes - 1:
                    survivors = []
                    if player_attacker.get_hp() > 0:
                        survivors.append(f"Maria (HP: {player_attacker.get_hp()})")
                    if player_support.get_hp() > 0:
                        survivors.append(f"Juana (HP: {player_support.get_hp()})")
                    
                    survivor_text = ", ".join(survivors) if survivors else "Nessuno"
                    
                    print(f"\n{'='*70}")
                    print(f"  {result}  |  Episode {e+1}/{episodes}")
                    print(f"{'='*70}")
                    print(f"  Attacker Reward: {total_reward_attacker:>6.0f}  |  Moves: {moves}")
                    print(f"  Support Reward:  {total_reward_support:>6.0f}  |  Epsilon: ATK={attacker_agent.epsilon:.3f}, SUP={supporter_agent.epsilon:.3f}")
                    print(f"  Sopravvissuti: {survivor_text}")
                    print(f"  Win Rate: {total_agent_wins}/{e+1} ({100*win_rate:.1f}%)")
                    print(f"  Vittorie agente: {total_agent_wins}, vittorie nemico: {total_enemy_wins}")
                    print(f"{'='*70}\n")
                
                break

        # Salva reward e mosse per l'episodio
        rewards_per_episode.append({
            'attacker': total_reward_attacker,
            'support': total_reward_support,
            'combined': total_reward_attacker + total_reward_support
        })
        agent_moves_per_episode.append(moves)
        
        # OTTIMIZZAZIONE: Se score calculation è disabilitato, usa 0
        action_scores.append({
            'attacker': np.mean(match_score_attacker) if match_score_attacker else 0,
            'support': np.mean(match_score_support) if match_score_support else 0,
            'combined': (np.mean(match_score_attacker) + np.mean(match_score_support)) / 2 if match_score_attacker and match_score_support else 0
        })
        
        # OTTIMIZZAZIONE: Stampa GPU stats ogni 100 episodi
        if (e + 1) % 100 == 0:
            print_gpu_stats()
    
    avg_reward_attacker = np.mean([r['attacker'] for r in rewards_per_episode])
    avg_reward_support = np.mean([r['support'] for r in rewards_per_episode])
    avg_reward_combined = np.mean([r['combined'] for r in rewards_per_episode])
    avg_moves = np.mean(agent_moves_per_episode)
    avg_score_attacker = np.mean([s['attacker'] for s in action_scores])
    avg_score_support = np.mean([s['support'] for s in action_scores])
    avg_score_combined = np.mean([s['combined'] for s in action_scores])
    
    print("\n=== TRAINING SUMMARY ===")
    print(f"Average reward (Attacker): {avg_reward_attacker:.2f}")
    print(f"Average reward (Support): {avg_reward_support:.2f}")
    print(f"Average reward (Combined): {avg_reward_combined:.2f}")
    print(f"Average moves: {avg_moves:.2f}")
    print(f"Average score (Attacker): {avg_score_attacker:.4f}")
    print(f"Average score (Support): {avg_score_support:.4f}")
    print(f"Average score (Combined): {avg_score_combined:.4f}")


    #if (e + 1) % 200 == 0:
    #    save_path = f"model_dqn_episode_{e + 1}"
    #    print(f"Saving model to {save_path}...")
    #    agent.save(save_path)
    
    attacker_agent.save("MODELLO_NO_LLM_ATTACKER") # save the agent model
    supporter_agent.save("MODELLO_NO_LLM_SUPPORT") # save the agent model

    #append_csv("reward_per_episode.csv", rewards_per_episode, "Reward")
    #append_csv("agent_wins.csv", agent_wins, "Wins")
    #append_csv("enemy_wins.csv", enemy_wins, "Wins")
    #append_csv("agent_moves.csv", agent_moves_per_episode, "Moves")
    #append_csv("success_rate.csv", success_rate, "Rate")

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores


# Plotting function
def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores):
    plt.figure(figsize=(8, 6))
    reward_attacker = [r['attacker'] for r in rewards]
    reward_support = [r['support'] for r in rewards]
    reward_combined = [r['combined'] for r in rewards]
    
    plt.plot(reward_attacker, label='Attacker Reward', color='red', alpha=0.7)
    plt.plot(reward_support, label='Support Reward', color='blue', alpha=0.7)
    plt.plot(reward_combined, label='Combined Reward', color='green', linewidth=2)
    
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    plt.savefig("Train_reward_DQN.png")

    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')

    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Train_cumulative_Win_DQN.png")

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Train_moves_DQN.png")

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Train_success_rate_DQN.png")

    plt.figure(figsize=(8, 6))
    attacker_scores = [s['attacker'] for s in action_scores]
    support_scores = [s['support'] for s in action_scores]
    combined_scores = [s['combined'] for s in action_scores]
    
    plt.plot(attacker_scores, label='Attacker Score', color='red', alpha=0.7)
    plt.plot(support_scores, label='Support Score', color='blue', alpha=0.7)
    plt.plot(combined_scores, label='Combined Score', color='green', linewidth=2)
    
    plt.title('Action Scores per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend()
    plt.savefig("Score_DQN_separated.png")



def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })

    df.to_csv('train_success_rate_model_dqn_1000.csv', index=False)


def append_csv(path, data, column_name):
    df = pd.DataFrame({
        "Episode": list(range(1, 1 + len(data))),
        column_name: data
    })
    df.to_csv(path, index=False)


def load_csv_series(filename, column):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df[column].tolist()
    return []


if __name__ == "__main__":
    # Spell offensivi (già esistenti)
    fire = Spell("Fire", 25, 600, "black")
    thunder = Spell("Thunder", 30, 700, "black")
    blizzard = Spell("Blizzard", 35, 800, "black")
    meteor = Spell("Meteor", 40, 1000, "black")

    # Spell curativi per ATTACKER
    cura = Spell("Cura", 32, 1500, "white")

    # Spell curativi per SUPPORT
    cura_support = Spell("Cura", 32, 1200, "white")  # Auto-cure
    cura_tot = Spell("Cura Tot", 30, 700, "white_tot")  # Cura entrambi
    splash = Spell("Splash", 18, 450, "white_tot")  # Cura entrambi (meno potente)
    cura_m = Spell("Cura M", 28, 1300, "white_m")  # Cura il mate
    cura_totm = Spell("Cura TotM", 36, 1700, "white_m")  # Cura di più il mate

    potion = Item("Potion", "potion", "Heals 50 HP", 50)
    hielixer = Item("MegaElixer", "elixir", "Fully restores party's HP/MP", 9999)
    grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

    # Train the agent - BATCH SIZE OTTIMIZZATO
    # - batch_size=128: bilanciato per apprendimento efficace
    # - Batch più piccoli = più aggiornamenti = apprendimento migliore
    # - 512 era troppo grande e causava overfitting
    rewards, agent_wins, enemy_wins, moves, success_rate, action_scores = train_dqn(
        episodes=10,
        batch_size=128  # ← RIDOTTO da 512 per miglior apprendimento
    )
    
    # Plot dei risultati
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores)
    
    # Esporta success rate
    export_success_rate(success_rate)
    
    print("\nTraining completato!")
    print("Grafici salvati:")
    print("   - Train_reward_DQN.png")
    print("   - Train_cumulative_Win_DQN.png")
    print("   - Train_moves_DQN.png")
    print("   - Train_success_rate_DQN.png")
    print("   - Score_DQN_separated.png")
    print("Modelli salvati:")
    print("   - MODELLO_NO_LLM_ATTACKER.h5")
    print("   - MODELLO_NO_LLM_SUPPORT.h5")
