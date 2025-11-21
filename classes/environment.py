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
        self.supporter_agent = None

    def set_supporter(self, agent):
        self.supporter_agent = agent

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

    def step(self, action):
        reward = 0
        agent_win = False
        enemy_win = False

        # human-readable move tracking for per-turn log
        main_move_desc = "No action"
        supp_move_desc = "No action"
        enemy_move_desc = "No action"


        # Agent choise
        if action == 0:
            for player in self.players:
                dmg = player.generate_damage()
                enemy = self.enemies[0]
                enemy.take_damage(dmg)
                reward += 25
            main_move_desc = "attack"

        elif action > 0 and action <= len(self.players[0].magic):
            player = self.players[0]
            spell = player.magic[action - 1]
            main_move_desc = spell.name
            if player.get_mp() >= spell.cost:
                magic_dmg = spell.generate_damage()
                player.reduce_mp(spell.cost)
                if spell.type == "white":
                    player.heal(magic_dmg)
                else:
                    enemy = self.enemies[0]
                    enemy.take_damage(magic_dmg)
                reward += 15

        # Main player's items (separate block so supporter presence doesn't skip it)
        if action > len(self.players[0].magic):
            item_index = action - len(self.players[0].magic) - 1
            player = self.players[0]
            if 0 <= item_index < len(player.items):
                item = player.items[item_index]["item"]
                main_move_desc = item.name
                if player.items[item_index]["quantity"] > 0:
                    player.items[item_index]["quantity"] -= 1
                    if item.type == "potion":
                        player.heal(item.prop)
                        reward += 15
                    elif item.type == "attack":
                        enemy = self.enemies[0]
                        enemy.take_damage(item.prop)
                        reward += 15
                    elif item.type == "elixir":
                        if player.hp <= 1000:
                            reward += 50
                        else:
                            reward += 5
                        player.hp = player.maxhp
                        player.mp = player.maxmp

        # Supporter (if present) acts after the main agent and before enemies
        if self.supporter_agent is not None and len(self.players) > 1:
            try:
                supp_action = self.supporter_agent.act(self)
            except Exception:
                supp_action = None

            if supp_action is not None:
                # build supporter move description
                supporter_player = self.players[1]
                if supp_action == 0:
                    supp_move_desc = "attack"
                elif supp_action > 0 and supp_action <= len(supporter_player.magic):
                    supp_move_desc = supporter_player.magic[supp_action - 1].name
                else:
                    idx = supp_action - len(supporter_player.magic) - 1
                    if 0 <= idx < len(supporter_player.items):
                        supp_move_desc = supporter_player.items[idx]["item"].name
                print(f"Supporter action chosen: {supp_action}")
                # Supporter actions follow the local indexing: 0 attack, 1..magic, then items
                if supp_action == 0:
                    dmg = supporter_player.generate_damage()
                    self.enemies[0].take_damage(dmg)
                    reward += 5
                    print(f"Supporter attacks for {dmg} dmg")

                elif supp_action > 0 and supp_action <= len(supporter_player.magic):
                    spell = supporter_player.magic[supp_action - 1]
                    if supporter_player.get_mp() >= spell.cost:
                        magic_dmg = spell.generate_damage()
                        supporter_player.reduce_mp(spell.cost)
                        if spell.type == "white":
                            # If spell name indicates it should heal mate (contains 'm' or 'M'), heal ally
                            if 'm' in spell.name.lower():
                                # heal the other player (index 0)
                                self.players[0].heal(magic_dmg)
                            elif spell.name=='Splash':
                                for p in self.players:
                                    p.heal(magic_dmg)
                            else:
                                supporter_player.heal(magic_dmg)
                        else:
                            self.enemies[0].take_damage(magic_dmg)
                        reward += 10

                elif supp_action > len(supporter_player.magic):
                    item_index = supp_action - len(supporter_player.magic) - 1
                    if item_index >= 0 and item_index < len(supporter_player.items):
                        if supporter_player.items[item_index]["quantity"] > 0:
                            supporter_player.items[item_index]["quantity"] -= 1
                            item = supporter_player.items[item_index]["item"]
                            if item.type == "potion":
                                supporter_player.heal(item.prop)
                                reward += 10
                                print(f"Supporter used potion on self (+{item.prop})")
                            elif item.type == "attack":
                                self.enemies[0].take_damage(item.prop)
                                reward += 10
                                print(f"Supporter used attack item for {item.prop} dmg")
                            elif item.type == "elixir":
                                supporter_player.hp = supporter_player.maxhp
                                supporter_player.mp = supporter_player.maxmp
                                reward += 10
                                print(f"Supporter used elixir: full restore")

        elif action > len(self.players[0].magic):
            item_index = action - len(self.players[0].magic) - 1
            player = self.players[0]
            item = player.items[item_index]["item"]
            if player.items[item_index]["quantity"] > 0:
                player.items[item_index]["quantity"] -= 1
                if item.type == "potion":
                    player.heal(item.prop)
                    reward += 15
                elif item.type == "attack":
                    enemy = self.enemies[0]
                    enemy.take_damage(item.prop)
                    reward += 15
                elif item.type == "elixir":
                    if player.hp <= 1000:
                        reward += 50
                    else:
                        reward += 5
                    player.hp = player.maxhp
                    player.mp = player.maxmp


        # Check for battle ended (log turn before returning)
        if self.enemies[0].get_hp() <= 0:
            self.done = True
            reward += 100
            agent_win = True
            p1hp = self.players[0].get_hp()
            p2hp = self.players[1].get_hp() if len(self.players) > 1 else None
            ehdp = self.enemies[0].get_hp()
            print(f"Turn log -> P1: HP={p1hp} Move={main_move_desc} | P2: HP={p2hp} Move={supp_move_desc} | Enemy: HP={ehdp} Move={enemy_move_desc}")
            return self.get_state(), reward, self.done, agent_win, enemy_win, "No action"

        # Enemy choise (Random)
        for enemy in self.enemies:
            enemy_choice = 'attack'

            # If the enemy has enough magic points, it can also choose magic.
            if enemy.get_mp() >= MIN_SPELL_COST:
                enemy_choice = random.choice(['attack', 'magic'])

            if enemy_choice == 'attack':
                target = random.choice(self.players)
                enemy_dmg = enemy.generate_damage()
                target.take_damage(enemy_dmg)
                enemy_move_desc = 'attack'

            elif enemy_choice == 'magic':
                spell, magic_dmg = enemy.choose_enemy_spell()
                enemy_choice = spell.name
                if enemy.get_mp() >= spell.cost:
                    enemy.reduce_mp(spell.cost)
                    if spell.type == "white":
                        enemy.heal(magic_dmg)
                        enemy_move_desc = spell.name
                    else:
                        target = random.choice(self.players)
                        target.take_damage(magic_dmg)
                        enemy_move_desc = spell.name

        #  Check for battle ended
        # print turn log before final returns
        p1hp = self.players[0].get_hp()
        p2hp = self.players[1].get_hp() if len(self.players) > 1 else None
        ehdp = self.enemies[0].get_hp()
        print(f"Turn log -> P1: HP={p1hp} Move={main_move_desc} | P2: HP={p2hp} Move={supp_move_desc} | Enemy: HP={ehdp} Move={enemy_move_desc}")

        if all(p.get_hp() <= 0 for p in self.players):
            self.done = True
            reward -= 100
            enemy_win = True

        return self.get_state(), reward, self.done, agent_win, enemy_win, enemy_choice

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


