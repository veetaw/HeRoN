import numpy as np
import matplotlib.pyplot as plt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.agent import DQNAgent
from classes.environment import BattleEnv
import pandas as pd
import re
import action_score as score

try:
    import lmstudio as lms
    lms.get_default_client(SERVER_API_HOST)

    def get_llm_response(input_text):
        with lms.Client() as client:
            model = client.llm.model("") # Helper model
            llm_response = model.respond(input_text)

        llm_response = str(llm_response)
        return llm_response

except ImportError:
    from openai import OpenAI
    client = OpenAI(base_url=SERVER_API_HOST, api_key="lm-studio")

    def get_llm_response(input_text):
        response_obj = client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": input_text}],
            temperature=0.7,
            max_tokens=100
        )
        return response_obj.choices[0].message.content


def map_llm_action_to_attacker_action(llm_response):
    match = re.search(r'\[(.*?)\]', llm_response)
    if match:
        action = match.group(1).strip().lower()
        if action == "attack":
            return 0
        elif action == "fire spell":
            return 1
        elif action == "thunder spell":
            return 2
        elif action == "blizzard spell":
            return 3
        elif action == "meteor spell":
            return 4
        elif action == "cura spell":
            return 5
        elif action == "potion":
            return 6
        elif action == "grenade":
            return 7
        elif action == "elixir" or action == "elixer":
            return 8
    return None

def map_llm_action_to_supporter_action(llm_response):
    match = re.search(r'\[(.*?)\]', llm_response)
    if match:
        action = match.group(1).strip().lower()
        if action == "attack":
            return 0
        elif action == "fire spell":
            return 1
        elif action == "cura spell":
            return 2
        elif action == "cura_tot":
            return 3
        elif action == "splash":
            return 4
        elif action == "cura_m":
            return 5
        elif action == "cura_totm":
            return 6
        elif action == "potion":
            return 7
        elif action == "grenade":
            return 8
        elif action == "elixir" or action == "elixer":
            return 9
    return None

# implementa map_llm_action_to_agent_action per il supporter
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

#aggiungere la nuova map action del supporter

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

#aggiungere il secondo player supporter
def train_dqn(episodes, batch_size=32):
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
    player2 = Person("Juana", 2300, 180, 100, 50, support_spells, player_items)
    enemy1 = Person("Antonio", 8000, 701, 525, 25, [fire, cura], [])

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    #NPC
    # al posto di env.state_size va messo len(env.get state (index)), idem per actiopn size
    attacker_agent = DQNAgent(
        env.get_state_size_of_player('Maria'), 
        env.get_action_size(0), 
        load_model_path
    )
    supporter_agent = DQNSupportAgent(
        env.get_state_size_of_player('Juana'), 
        env.get_action_size(1), 
        load_model_path
    )


    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []
    allucination = 0
    total_agent_wins = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        moves = 0
        match_score = []
        last_enemy_move = "No action"
        score.reset_quantity()
        while not done:
            #  Description of environment and Helper action #
            game_description_attacker = env.describe_game_state_attacker(last_enemy_move)
            #  prompt per l'attacker
            input_text_attacker = "You are a game asstitant for player." \
                                "The battle involves with 2 players (attacker and supporter) and an enemy. Both player and enemy are characterised by:" \
                                "HP (health point), MP (magic point), Attack points, Defence points, Available magic points " \
                                "and Available items. " \
                                "For spells, the use of MP is necessary, while items have limited availability. " \
                                "Now you are the attacker, the player can have a maximum of 2600 HP and 120 MP, while the enemy 8000 HP and 701 MP. " \
                                f"Given the game state '{game_description_attacker}', what is the next action to take? " \
                                "Write only the chosen action in square brackets and " \
                                "explain your reasoning briefly, max 50 words. /no_think"
            game_description_supporter = env.describe_game_state_supporter(last_enemy_move)
            # prompt per il supporter
            input_text_supporter = "You are a game asstitant for player." \
                                "The battle involves with 2 players (attacker and supporter) and an enemy. Both player and enemy are characterised by:" \
                                "HP (health point), MP (magic point), Attack points, Defence points, Available magic points " \
                                "and Available items. " \
                                "For spells, the use of MP is necessary, while items have limited availability. " \
                                "Now you are the helper, the player can have a maximum of 2300 HP and 180 MP, your mate can have a maximum of 2600 HP and 120 MP, while the enemy 8000 HP and 701 MP. " \
                                f"Given the game state '{game_description_supporter}', what is the next action to take? " \
                                "Write only the chosen action in square brackets and " \
                                "explain your reasoning briefly, max 50 words. /no_think"

            llm_response_attacker = get_llm_response(input_text_attacker)
            llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
            print(f"LLM response: {llm_response}")

            # Mapping LLM action to RL agent with action score calculation #
            action = map_llm_action_to_attacker_action(llm_response)
            #aggiungere il mapping per il supporter

            if action != None:
                match = re.search(r'\[(.*?)\]', llm_response)
                match = match.group(1).strip().lower()
                if match == "elixer":
                    match = "elixir"
                total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
                match_score.append(round(total_score.get(match), 2))
            else:
                action = agent.act(state, env)
                match = map_action_attack(action)
                total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
                match_score.append(round(total_score.get(match), 2))
                allucination += 1

            # Execution of RL action #
            next_state, reward, done, a_win, e_win, last_enemy_move = env.step(action)
            score.updage_quantity(match, players[0].get_mp())
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves += 1
            # dovrebbe andare qui l'Helper (?)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)

            if done:
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, Moves: {moves}, Epsilon: {agent.epsilon}")
                if a_win:
                    agent_wins.append(1)
                    enemy_wins.append(0)
                    total_agent_wins += 1
                else:
                    agent_wins.append(0)
                    enemy_wins.append(1)

                success_rate.append(total_agent_wins / (e + 1))
                print("Vittorie agente: ", agent_wins.count(1), " Vittorie nemico: ", enemy_wins.count(1))
        rewards_per_episode.append(total_reward)
        agent_moves_per_episode.append(moves)
        action_scores.append(np.mean(match_score))

    agent.save("") # save the agent model
    print("Average rewards: ", np.mean(rewards_per_episode))
    print("Average moves: ", np.mean(agent_moves_per_episode))
    print("Average move score: ", np.mean(action_scores))
    print("Hallucinations: ", allucination)

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores


def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig("Reward_llama_1000.png")
    #plt.show()

    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')
    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Cumulative_Win_llama_1000.png")
    #plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Moves_llama_1000.png")
    #plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Success_rate_llama_1000.png")
    #plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(match_score)
    plt.title('Score moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Score')
    plt.savefig("Score_llama_1000.png")
    #plt.show()


def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })
    df.to_csv('success_rate_model_llama_1000.csv', index=False)


if __name__ == "__main__":
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

    # Train the agent
    rewards, agent_wins, enemy_wins, moves, success_rate, action_scores = train_dqn(episodes=10)
    
    # Plot dei risultati
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores)
    export_success_rate(success_rate)
