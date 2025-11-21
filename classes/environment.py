import numpy as np
import random
from classes.magic import Spell
from classes.inventory import Item

# Spells and items setup
fire = Spell("Fire", 25, 600, "black")
thunder = Spell("Thunder", 30, 700, "black")
blizzard = Spell("Blizzard", 35, 800, "black")
meteor = Spell("Meteor", 40, 1000, "black")
cura = Spell("Cura", 32, 1500, "white")
MIN_SPELL_COST =  25

potion = Item("Potion", "potion", "Heals 50 HP", 50)
hielixer = Item("MegaElixer", "elixir", "Fully restores party's HP/MP", 9999)
grenade = Item("Grenade", "attack", "Deals 500 damage", 500)


# Environment setup
class BattleEnv:
    def __init__(self, players, enemies):
        self.players = players
        self.enemies = enemies
        self.state_size = len(self.get_state())
        self.action_size = self.get_action_size()
        self.done = False

    def get_state(self):
        state = []
        # Add player and enemy stats (HP, MP, spells, items)
        for player in self.players:
            state.extend([player.get_hp(), player.get_mp()])
            for spell in player.magic:
                state.append(1 if player.get_mp() >= spell.cost else 0)  # Can cast spell
            for item in player.items:
                state.append(item["quantity"])  # Quantity of items left
        if len(self.enemies) == 0:
            state.extend([0, 0])
        for enemy in self.enemies:
            state.extend([enemy.get_hp(), enemy.get_mp()])
        print(f"State: {state}, Length: {len(state)}")
        return np.array(state)

    def get_action_size(self):
        # Total actions: attack (1), spells (len(player.magic)), items (len(player.items))
        actions = 1  # Attack
        for player in self.players:
            actions += len(player.magic)  # Each spell is a separate action
            actions += len(player.items)  # Each item is a separate action
        return actions

    def get_valid_actions(self):
        valid_actions = [0]  # Attack always valid
        player = self.players[0]
        # Spells
        for idx, spell in enumerate(player.magic):
            if player.get_mp() >= spell.cost:
                valid_actions.append(idx + 1)
        # Items
        for idx, item in enumerate(player.items):
            if item["quantity"] > 0:
                # Items start after magic actions
                valid_actions.append(len(player.magic) + 1 + idx)
        return valid_actions

    def reset(self):
        self.done = False
        for player in self.players:
            player.hp = player.maxhp
            player.mp = player.maxmp
        for enemy in self.enemies:
            enemy.hp = enemy.maxhp
            enemy.mp = enemy.maxmp

        return self.get_state()

    def step(self, actions):
        if not isinstance(actions, list):
            actions = [actions]
        
        rewards = []
        agent_win = False
        enemy_win = False
        player_moves = []

        for player_idx, action in enumerate(actions):
            if player_idx >= len(self.players):
                continue
            
            player = self.players[player_idx]
            move_desc = "No action"
            player_reward = 0
            
            if action == 0:
                dmg = player.generate_damage()
                enemy = self.enemies[0]
                enemy.take_damage(dmg)
                if player_idx == 0:
                    player_reward = 25
                else:
                    player_reward = 10
                move_desc = "attack"

            elif action > 0 and action <= len(player.magic):
                spell = player.magic[action - 1]
                if player.get_mp() >= spell.cost:
                    magic_dmg = spell.generate_damage()
                    player.reduce_mp(spell.cost)
                    if spell.type == "white":
                        if player_idx == 0:
                            hp_before = player.get_hp()
                            player.heal(magic_dmg)
                            hp_gained = player.get_hp() - hp_before
                            player_reward = max(5, hp_gained // 50)
                        else:
                            if spell.name.lower().endswith('m'):
                                target_player = self.players[0]
                                hp_before = target_player.get_hp()
                                target_player.heal(magic_dmg)
                                hp_gained = target_player.get_hp() - hp_before
                                player_reward = max(10, hp_gained // 30)
                            elif 'splash' in spell.name.lower():
                                total_hp_gained = 0
                                for p in self.players:
                                    hp_before = p.get_hp()
                                    p.heal(magic_dmg)
                                    total_hp_gained += (p.get_hp() - hp_before)
                                player_reward = max(15, total_hp_gained // 40)
                            else:
                                hp_before = player.get_hp()
                                player.heal(magic_dmg)
                                hp_gained = player.get_hp() - hp_before
                                player_reward = max(8, hp_gained // 40)
                    else:
                        enemy = self.enemies[0]
                        enemy.take_damage(magic_dmg)
                        if player_idx == 0:
                            player_reward = 20
                        else:
                            player_reward = 10
                    move_desc = spell.name

            elif action > len(player.magic):
                item_index = action - len(player.magic) - 1
                if 0 <= item_index < len(player.items):
                    if player.items[item_index]["quantity"] > 0:
                        player.items[item_index]["quantity"] -= 1
                        item = player.items[item_index]["item"]
                        move_desc = item.name
                        if item.type == "potion":
                            if player_idx == 0:
                                hp_before = player.get_hp()
                                player.heal(item.prop)
                                hp_gained = player.get_hp() - hp_before
                                player_reward = max(5, hp_gained // 10)
                            else:
                                target_player = self.players[0]
                                hp_before = target_player.get_hp()
                                target_player.heal(item.prop)
                                hp_gained = target_player.get_hp() - hp_before
                                player_reward = max(8, hp_gained // 8)
                        elif item.type == "attack":
                            enemy = self.enemies[0]
                            enemy.take_damage(item.prop)
                            if player_idx == 0:
                                player_reward = 20
                            else:
                                player_reward = 10
                        elif item.type == "elixir":
                            if player.hp <= 1000:
                                player_reward = 50
                            else:
                                player_reward = 5
                            player.hp = player.maxhp
                            player.mp = player.maxmp
            
            player_moves.append(move_desc)
            rewards.append(player_reward)

        if self.enemies[0].get_hp() <= 0:
            self.done = True
            for i in range(len(rewards)):
                rewards[i] += 100
            agent_win = True
            p1hp = self.players[0].get_hp()
            p2hp = self.players[1].get_hp() if len(self.players) > 1 else None
            enemy_hp = self.enemies[0].get_hp()
            main_move_desc = player_moves[0] if len(player_moves) > 0 else "No action"
            supp_move_desc = player_moves[1] if len(player_moves) > 1 else "No action"
            print(f"Turn log -> P1: HP={p1hp} Move={main_move_desc} | P2: HP={p2hp} Move={supp_move_desc} | Enemy: HP={enemy_hp} Move=No action")
            total_reward = sum(rewards) if len(rewards) > 0 else 0
            return self.get_state(), total_reward, self.done, agent_win, enemy_win, "No action"

        enemy_move_desc = "No action"
        for enemy in self.enemies:
            enemy_choice = 'attack'

            if enemy.get_mp() >= MIN_SPELL_COST:
                enemy_choice = random.choice(['attack', 'magic'])

            if enemy_choice == 'attack':
                target = random.choice(self.players)
                enemy_dmg = enemy.generate_damage()
                target.take_damage(enemy_dmg)
                enemy_move_desc = "attack"

            elif enemy_choice == 'magic':
                spell, magic_dmg = enemy.choose_enemy_spell()
                enemy_choice = spell.name
                if enemy.get_mp() >= spell.cost:
                    enemy.reduce_mp(spell.cost)
                    if spell.type == "white":
                        enemy.heal(magic_dmg)
                    else:
                        target = random.choice(self.players)
                        target.take_damage(magic_dmg)
                    enemy_move_desc = spell.name

        if all(p.get_hp() <= 0 for p in self.players):
            self.done = True
            for i in range(len(rewards)):
                rewards[i] -= 100
            enemy_win = True

        p1hp = self.players[0].get_hp()
        p2hp = self.players[1].get_hp() if len(self.players) > 1 else None
        enemy_hp = self.enemies[0].get_hp()
        main_move_desc = player_moves[0] if len(player_moves) > 0 else "No action"
        supp_move_desc = player_moves[1] if len(player_moves) > 1 else "No action"
        print(f"Turn log -> P1: HP={p1hp} Move={main_move_desc} | P2: HP={p2hp} Move={supp_move_desc} | Enemy: HP={enemy_hp} Move={enemy_move_desc}")
        total_reward = sum(rewards) if len(rewards) > 0 else 0
        return self.get_state(), total_reward, self.done, agent_win, enemy_win, enemy_choice

    def describe_game_state(self, last_enemy_move):
        state_description = ""

        for player in self.players:
            state_description += f"Player has {player.get_hp()} Health Points (hp) and {player.get_mp()} Magic Points (mp). "

        for enemy in self.enemies:
            state_description += f"Enemy has {enemy.get_hp()} Health Points (hp) and {enemy.get_mp()} Magic Points (mp). "

        actions_description = "Available actions: [attack] deals 300 enemy's hp and removes 0 player's mp; "

        player = self.players[0]

        if player.get_mp() >= fire.cost:
            fire_spell = "[fire spell] deals 600 enemy's hp and removes 25 player's mp; "
            actions_description += fire_spell
        if player.get_mp() >= thunder.cost:
            thunder_spell = "[thunder spell] deals 700 enemy's hp and removes 30 player's mp; "
            actions_description += thunder_spell
        if player.get_mp() >= blizzard.cost:
            blizzard_spell = "[blizzard spell] deals 800 enemy's hp and removes 35 player's mp; "
            actions_description += blizzard_spell
        if player.get_mp() >= meteor.cost:
            meteor_spell = "[meteor spell] deals 1000 enemy's hp and removes 40 player's mp; "
            actions_description += meteor_spell
        if player.get_mp() >= cura.cost:
            cura_spell = "[cura spell] heals 1500 player's hp and removes 32 player's mp; "
            actions_description += cura_spell
        if player.items[0]["quantity"] > 0:
            potion = f"[potion] heals 50 player's hp and there are {player.items[0]['quantity']}; "
            actions_description += potion
        if player.items[1]["quantity"] > 0:
            grenade = f"[grenade] deals 500 enemy's hp and there are {player.items[1]['quantity']}; "
            actions_description += grenade
        if player.items[2]["quantity"] > 0:
            elixer = f"[elixir] fully restores player's hp and mp and there are {player.items[2]['quantity']}. "
            actions_description += elixer

        last_move_description = f"Last enemy move was [{last_enemy_move}]."

        game_description = state_description + actions_description + last_move_description
        return game_description


