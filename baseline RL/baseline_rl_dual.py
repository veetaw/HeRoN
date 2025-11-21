import numpy as np
import matplotlib.pyplot as plt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.environment import BattleEnv
from classes.agent import DQNAgent
import pandas as pd

def map_action(action):
    if action == 0:
        return "attack"
    elif action == 1:
        return "fire spell"
    elif action == 2:
        return "thunder spell"
    elif action == 3:
        return "blizzard spell"
    elif action == 4:
        return "meteor spell"
    elif action == 5:
        return "cura spell"
    elif action == 6:
        return "potion"
    elif action == 7:
        return "grenade"
    elif action == 8:
        return "elixir"
    return None

def train_dual_agents(episodes, batch_size=32, load_model_path_1=None, load_model_path_2=None):
    fire = Spell("Fire", 25, 600, "black")
    thunder = Spell("Thunder", 30, 700, "black")
    blizzard = Spell("Blizzard", 35, 800, "black")
    meteor = Spell("Meteor", 40, 1000, "black")
    cura = Spell("Cura", 32, 1500, "white")

    potion = Item("Potion", "potion", "Heals 50 HP", 50)
    hielixer = Item("MegaElixer", "elixer", "Fully restores party's HP/MP", 9999)
    grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

    player_spells_1 = [fire, thunder, blizzard, meteor, cura]
    player_items_1 = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                      {"item": hielixer, "quantity": 1}]
    player1 = Person("Valos", 3260, 132, 300, 34, player_spells_1, player_items_1)

    player_spells_2 = [fire, cura]
    player_items_2 = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                      {"item": hielixer, "quantity": 1}]
    player2 = Person("Cristo", 2800, 200, 120, 20, player_spells_2, player_items_2)

    enemy1 = Person("Magus", 4000, 701, 525, 25, [fire, cura], [])

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)

    agent1 = DQNAgent(env.state_size, env.action_size, load_model_path_1)
    agent2 = DQNAgent(env.state_size, env.action_size, load_model_path_2)

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []

    total_agent_wins = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        moves = 0

        while not done:
            action1 = agent1.act(state, env)
            action2 = agent2.act(state, env)
            
            next_state, reward, done, a_win, e_win, enemy_choice = env.step([action1, action2])

            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            
            agent1.remember(state, action1, reward, next_state, done)
            agent2.remember(state, action2, reward, next_state, done)
            
            state = next_state
            moves += 1

            if len(agent1.memory) > batch_size:
                agent1.replay(batch_size, env)
            if len(agent2.memory) > batch_size:
                agent2.replay(batch_size, env)

            if done:
                if a_win:
                    agent_wins.append(1)
                    enemy_wins.append(0)
                    total_agent_wins += 1
                else:
                    agent_wins.append(0)
                    enemy_wins.append(1)

                success_rate.append(total_agent_wins / (e + 1))
                print(f"Episode {e+1}/{episodes} - Wins: {agent_wins.count(1)} - Losses: {enemy_wins.count(1)}")
        
        rewards_per_episode.append(total_reward)
        agent_moves_per_episode.append(moves)

    print(f"Average rewards: {np.mean(rewards_per_episode)}")
    print(f"Average moves: {np.mean(agent_moves_per_episode)}")

    agent1.save("agent1_dual")
    agent2.save("agent2_dual")

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate

def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig("Train_reward_DQN_dual.png")

    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')

    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Train_cumulative_Win_DQN_dual.png")

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Train_moves_DQN_dual.png")

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Train_success_rate_DQN_dual.png")

if __name__ == "__main__":
    rewards, agent_wins, enemy_wins, moves, success_rate = train_dual_agents(episodes=1000)
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate)
