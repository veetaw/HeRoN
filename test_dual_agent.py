import numpy as np
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.environment import BattleEnv
from classes.agent import DQNAgent

fire = Spell("Fire", 25, 600, "black")
cura = Spell("Cura", 32, 1500, "white")
potion = Item("Potion", "potion", "Heals 50 HP", 50)

player_spells_1 = [fire, cura]
player_items_1 = [{"item": potion, "quantity": 2}]
player1 = Person("Attacker", 1000, 100, 50, 10, player_spells_1, player_items_1)

player_spells_2 = [cura]
player_items_2 = [{"item": potion, "quantity": 2}]
player2 = Person("Healer", 800, 120, 30, 10, player_spells_2, player_items_2)

enemy1 = Person("Enemy", 1500, 100, 40, 10, [fire], [])

players = [player1, player2]
enemies = [enemy1]

env = BattleEnv(players, enemies)

agent1 = DQNAgent(env.state_size, env.action_size, None)
agent2 = DQNAgent(env.state_size, env.action_size, None)

state = env.reset()
state = np.reshape(state, [1, env.state_size])

for i in range(5):
    action1 = agent1.act(state, env)
    action2 = agent2.act(state, env)
    
    print(f"Turn {i+1}: Agent1 action={action1}, Agent2 action={action2}")
    
    next_state, reward, done, a_win, e_win, _ = env.step([action1, action2])
    
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Player1 HP: {players[0].get_hp()}, Player2 HP: {players[1].get_hp()}, Enemy HP: {enemies[0].get_hp()}")
    
    if done:
        if a_win:
            print("Agents win!")
        else:
            print("Enemy wins!")
        break
    
    state = np.reshape(next_state, [1, env.state_size])

print("Test completed successfully!")
