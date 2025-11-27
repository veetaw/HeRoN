import numpy as np
import random
from classes.magic import Spell
from classes.inventory import Item

# Spells and items setup
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
MIN_SPELL_COST =  25

potion = Item("Potion", "potion", "Heals 50 HP", 50)
hielixer = Item("MegaElixer", "elixir", "Fully restores party's HP/MP", 9999)
grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

ATTACKER_INDEX = 0
SUPPORT_INDEX = 1


# Environment setup
class BattleEnv:
    def __init__(self, players, enemies):
        """
        players: lista di Person
        enemies: lista di Person
        """
        self.players = players
        self.enemies = enemies
        self.original_player_data = []
        for player in players:
            self.original_player_data.append({
                'name': player.name,
                'magic': player.magic.copy(),  # Copia gli spell originali
                'items': [{"item": item["item"], "quantity": item["quantity"]} for item in player.items]  # Copia items
            })
        
        self.state_size = len(self.get_state())
        self.action_size = self.get_action_size(0)
        self.done = False


    def get_state_size_of_player(self, player_name):
        """Ritorna la dimensione dello stato per un player specifico"""
        player = None
        for p in self.players:
            if p.name == player_name:
                player = p
                break
        
        if player is None:
            raise ValueError(f"Player {player_name} not found")
        
        # Stato = [HP, MP, spell1, spell2, ..., spellN, item1, item2, item3]
        num_spells = len(player.magic)
        num_items = len(player.items)
        state_size = 2 + num_spells + num_items  # HP + MP + spells + items
        
        print(f"DEBUG: Player {player_name} - Spells: {num_spells}, Items: {num_items}, State size: {state_size}")
        
        return state_size

    def get_state(self):
        """
        crea un'array contentente tutto lo stato del gioco.
        lo stato è composto da:
            per ogni player:
                hp,
                mp,
                1 per ogni spell se il player ha abbastanza mp per lanciarla, 0 altrimenti
                per ogni item la quantità disponibile, tipo 2 granate
            poi per l'enemy:
                hp,
                mp
        """
        state = {}
        state['enemies'] = []
        # Add player and enemy stats (HP, MP, spells, items)
        for player in self.players:
            state[player.name] = []

            state[player.name].extend([player.get_hp(), player.get_mp()])
            for spell in player.magic:
                state[player.name].append(1 if player.get_mp() >= spell.cost else 0)  # Can cast spell
            for item in player.items:
                state[player.name].append(item["quantity"])  # Quantity of items left
        
        if len(self.enemies) == 0:
            state["enemies"] = [0, 0]
        for enemy in self.enemies:
            state["enemies"].extend([enemy.get_hp(), enemy.get_mp()])
        return state

    def get_action_size(self, player_index):
        """
        calcola il numero totale di azioni disponibili nel gioco:
        somma 1 (attacco sempre disponibile) + spell + items

        probabilmente non va bene per due players, dovrebbe ritornare una lista di liste, o meglio ancora una mappa
        """
        # Total actions: attack (1), spells (len(player.magic)), items (len(player.items))
        actions = 1  # Attack
        actions += len(self.players[player_index].magic)  # Each spell is a separate action
        actions += len(self.players[player_index].items)  # Each item is a separate action
        return actions

    def get_valid_actions(self, player_index):
        """
        NON FUNZIONA CON DUE PLAYERS

        Questa funzione restituisce una lista di azioni valide (eseguibili) in base allo stato corrente del gioco:

        Spell: per ogni spell solo se il player ha abbastanza MP
        Item: per ogni item solo se la quantità è > 0
        Attack: sempre valido

        """
        player = self.players[player_index]
        valid_actions = []
        
        # ✅ Se il player è morto, nessuna azione è valida
        if player.get_hp() <= 0:
            return []  # Nessuna azione possibile
        
        # Action 0: Attack (sempre valida se vivo)
        valid_actions.append(0)
        
        # Spell (actions 1 to len(magic))
        for i, spell in enumerate(player.magic):
            action_index = i + 1
            # ✅ Controlla se ha abbastanza MP
            if player.get_mp() >= spell.cost:
                valid_actions.append(action_index)
        
        # Items (actions after spells)
        item_start_index = len(player.magic) + 1
        for i, item_data in enumerate(player.items):
            action_index = item_start_index + i
            # ✅ Controlla se ha quantità disponibile
            if item_data["quantity"] > 0:
                valid_actions.append(action_index)
        
        return valid_actions

    def reset(self):
        """Reset del gioco ripristinando HP/MP e gli spell/items originali"""
        self.done = False
        
        for i, player in enumerate(self.players):
            player.hp = player.maxhp
            player.mp = player.maxmp
            
            # Ripristina gli spell originali
            original_data = self.original_player_data[i]
            player.magic = original_data['magic'].copy()
            
            # Ripristina le quantità degli items
            player.items = [
                {"item": item["item"], "quantity": item["quantity"]} 
                for item in original_data['items']
            ]
        
        # Reset nemici
        for enemy in self.enemies:
            enemy.hp = enemy.maxhp
            enemy.mp = enemy.maxmp

        return self.get_state()

    def perform_action(self, player_index, action):
        player = self.players[player_index]
        reward = 0


        if player.get_hp() <= 0:
            return 0

        if action == 0:
            dmg = player.generate_damage()
            enemy = self.enemies[0]
            enemy.take_damage(dmg)
            reward += 25

        # vedo se index azione è tra 0 e len magic, allora è una spell
        elif action > 0 and action <= len(player.magic) and player_index == ATTACKER_INDEX:
            spell = player.magic[action - 1]
            if player.get_mp() >= spell.cost:
                magic_dmg = spell.generate_damage()
                player.reduce_mp(spell.cost)
                if spell.type == "white":
                    player.heal(magic_dmg)
                else:
                    enemy = self.enemies[0]
                    enemy.take_damage(magic_dmg)
                reward += 15

        elif action > 0 and action <= len(player.magic) and player_index == SUPPORT_INDEX:
            spell = player.magic[action - 1]
            mate = self.players[ATTACKER_INDEX]
            if player.get_mp() >= spell.cost:
                magic_dmg = spell.generate_damage()
                player.reduce_mp(spell.cost)
                
                if spell.type.startswith("white"):
                    if "_" in spell.type:
                        dest = spell.type.split("white_")[1]
                    else:
                        dest = ""
                    
                    if dest == "" or dest is None:
                        if (player.hp + magic_dmg) > player.maxhp:
                            reward -= 5
                        else:
                            reward += 10
                        player.heal(magic_dmg)
                        
                    elif dest == "m":
                        if (mate.hp + magic_dmg) > mate.maxhp:
                            reward -= 5
                        else:
                            if mate.hp < 0.3 * mate.maxhp:
                                reward += 25
                            else:
                                reward += 15
                        mate.heal(magic_dmg)
                        
                    elif dest == "tot":
                        overheal_self = max(0, (player.hp + magic_dmg) - player.maxhp)
                        player.heal(magic_dmg)
                        
                        overheal_mate = max(0, (mate.hp + magic_dmg) - mate.maxhp)
                        mate.heal(magic_dmg)
                        
                        if overheal_self > magic_dmg * 0.5 or overheal_mate > magic_dmg * 0.5:
                            reward -= 3
                        else:
                            reward += 12
                    
                    else:
                        print(f"Warning: Unknown spell dest '{dest}', defaulting to self-heal")
                        player.heal(magic_dmg)
                        reward += 5

                else:
                    enemy = self.enemies[0]
                    enemy.take_damage(magic_dmg)
                    reward += 10

        elif action > len(player.magic):
            item_index = action - len(player.magic) - 1
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

        return reward

    def enemy_turn(self):
        import random
        
        enemy = self.enemies[0]
        
        if enemy.get_hp() <= 0:
            return None
        
        alive_players = [p for p in self.players if p.get_hp() > 0]
        
        if not alive_players:
            return None
        
        enemy_choice = 'attack'
        
        # If the enemy has enough magic points, it can also choose magic
        if enemy.get_mp() >= MIN_SPELL_COST:
            enemy_choice = random.choice(['attack', 'magic'])
        
        if enemy_choice == 'attack':
            target = random.choice(alive_players)
            damage = enemy.generate_damage()
            target.take_damage(damage)
            print(f"Enemy {enemy.name} attacks {target.name} for {damage} dmg! (HP: {target.get_hp()}/{target.maxhp})")
            
        elif enemy_choice == 'magic':
            spell, magic_dmg = enemy.choose_enemy_spell()
            enemy_choice = spell.name
            
            if enemy.get_mp() >= spell.cost:
                enemy.reduce_mp(spell.cost)
                
                if spell.type == "white":
                    enemy.heal(magic_dmg)
                    print(f"Enemy {enemy.name} casts {spell.name} and heals for {magic_dmg} HP! (HP: {enemy.get_hp()}/{enemy.maxhp})")
                else:
                    target = random.choice(alive_players)
                    target.take_damage(magic_dmg)
                    print(f"Enemy {enemy.name} casts {spell.name} on {target.name} for {magic_dmg} dmg! (HP: {target.get_hp()}/{target.maxhp})")
        
        return enemy_choice


        
    def step(self, attacker_action, support_action):
        reward_attacker = 0
        reward_support = 0
        
        # TODO: sta cosa ha senso?
        if attacker_action is not None:
            valid_attacker = self.get_valid_actions(ATTACKER_INDEX)

            # fallback su ATTACK
            if attacker_action not in valid_attacker:
                attacker_action = 0

            reward_attacker = self.perform_action(ATTACKER_INDEX, attacker_action)

        if support_action is not None:
            valid_support = self.get_valid_actions(SUPPORT_INDEX)

            # fallback su ATTACK
            if support_action not in valid_support:
                support_action = 0

            reward_support = self.perform_action(SUPPORT_INDEX, support_action)
                
        if self.players[ATTACKER_INDEX].get_hp() <= 0:
            reward_attacker -= 100
            reward_support -= 50
        
        if self.players[SUPPORT_INDEX].get_hp() <= 0:
            reward_support -= 100
            reward_attacker -= 30
        
        enemy_action = self.enemy_turn()
        
        a_win = None
        
        if self.players[ATTACKER_INDEX].get_hp() <= 0 and self.players[SUPPORT_INDEX].get_hp() <= 0:
            self.done = True
            a_win = False
        
        elif all(enemy.get_hp() <= 0 for enemy in self.enemies):
            self.done = True
            a_win = True
            reward_attacker += 100
            reward_support += 100
        
        next_state = self.get_state()
        return next_state, reward_attacker, reward_support, self.done, a_win, enemy_action, None
    

    def describe_game_state_attacker(self, last_enemy_move):
        """
        descrive lo stato del game per l'LLM, attualmente non va bene per due giocatori.

        """
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


    def describe_game_state_supporter(self, last_enemy_move):
        """
        descrive lo stato del game per l'LLM, attualmente non va bene per due giocatori.

        """
        state_description = ""

        for player in self.players:
            state_description += f"Player has {player.get_hp()} Health Points (hp) and {player.get_mp()} Magic Points (mp). "

        for enemy in self.enemies:
            state_description += f"Enemy has {enemy.get_hp()} Health Points (hp) and {enemy.get_mp()} Magic Points (mp). "

        actions_description = "Available actions: [attack] deals 300 enemy's hp and removes 0 player's mp; "

        player = self.players[1]

        if player.get_mp() >= fire.cost:
            fire_spell = "[fire spell] deals 600 enemy's hp and removes 25 player's mp; "
            actions_description += fire_spell
        if player.get_mp() >= cura_support.cost:
            cura_support_spell = "[cura spell] It heals 500 player's hp and removes 32 player's mp; "
            actions_description += cura_support_spell
        if player.get_mp() >= cura_tot.cost:
            cura_tot_spell = "[cura_tot] Heals 700 HP from both players and removes 30 MP from the player; "
            actions_description += cura_tot_spell
        if player.get_mp() >= splash.cost:
            splash_spell = "[splash] Heals 450 HP from both players and removes 18 MP from the player; "
            actions_description += splash_spell
        if player.get_mp() >= cura_m.cost:
            cura_m_spell = "[cura_m] heals 1300 mate's hp and removes 28 player's mp; "
            actions_description += cura_m_spell
        if player.get_mp() >= cura_totm.cost:
            cura_totm_spell = "[cura_totm] heals 1700 mate's hp and removes 36 player's mp; "
            actions_description += cura_totm_spell
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
