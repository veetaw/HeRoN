# Dictionary of actions
attacker_actions = {
    'attack': {'damage': 300, 'mp_cost': 0, 'heal': 0, 'quantity': 1},
    'fire spell': {'damage': 600, 'mp_cost': 25, 'heal': 0, 'quantity': 1},
    'thunder spell': {'damage': 700, 'mp_cost': 30, 'heal': 0, 'quantity': 1},
    'blizzard spell': {'damage': 800, 'mp_cost': 35, 'heal': 0, 'quantity': 1},
    'meteor spell': {'damage': 1000, 'mp_cost': 40, 'heal': 0, 'quantity': 1},
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1500, 'quantity': 1},  # Auto-cure
    'potion': {'damage': 0, 'mp_cost': 0, 'heal': 50, 'mp_heal': 0, "quantity": 3},
    'grenade': {'damage': 500, 'mp_cost': 0, 'heal': 0, 'quantity': 2},
    'elixir': {'damage': 0, 'mp_cost': 0, 'heal': 3260, 'mp_heal': 132, 'quantity': 1}
}
support_actions = {
    'attack': {'damage': 250, 'mp_cost': 0, 'heal': 0, 'quantity': 1},
    'fire spell': {'damage': 500, 'mp_cost': 25, 'heal': 0, 'quantity': 1},

    # Cure e supporto
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1200, 'quantity': 1},  # Auto-cure
    'cura_tot': {'damage': 0, 'mp_cost': 30, 'heal': 700, 'quantity': 1},     # Cura entrambi
    'splash': {'damage': 0, 'mp_cost': 18, 'heal': 450, 'quantity': 1},       # Cura entrambi ma meno
    'cura_m': {'damage': 0, 'mp_cost': 28, 'heal': 1300, 'quantity': 1},      # Cura il mate
    'cura_totm': {'damage': 0, 'mp_cost': 36, 'heal': 1700, 'quantity': 1},   # Cura di più il mate 
    # Oggetti
    'potion': {'damage': 0, 'mp_cost': 0, 'heal': 50, 'mp_heal': 0, 'quantity': 3},
    'grenade': {'damage': 400, 'mp_cost': 0, 'heal': 0, 'quantity': 2},
    'elixir': {'damage': 0, 'mp_cost': 0, 'heal': 3000, 'mp_heal': 150, 'quantity': 1}
}

def update_quantity(action_name, current_mp, player_index=0):
    """
    Aggiorna le quantità di items/MP dopo un'azione.
    
    Args:
        action_name: Nome dell'azione eseguita (es: "fire spell", "potion")
        current_mp: MP correnti del player dopo l'azione
        player_index: 0 per attacker, 1 per support
    """
    # Seleziona il dizionario corretto
    actions_dict = attacker_actions if player_index == 0 else support_actions
    
    if action_name not in actions_dict:
        print(f"Warning: Action '{action_name}' not found for player {player_index}")
        return
    
    action = actions_dict[action_name]
    
    # Aggiorna quantità items (se è un item consumabile)
    if action_name in ['potion', 'grenade', 'elixir']:
        if action['quantity'] > 0:
            action['quantity'] -= 1
    
    # (Opzionale) Sincronizza MP nel dizionario
    # Questo è utile solo se vuoi tenere traccia degli MP anche nel dizionario
    # Ma normalmente gli MP vengono gestiti dall'oggetto Person
    
    print(f"[Player {player_index}] Updated {action_name}: quantity={action['quantity']}, MP={current_mp}")


def reset_quantities():
    """Reset delle quantità all'inizio di ogni episodio"""
    # Attacker
    attacker_actions['potion']['quantity'] = 3
    attacker_actions['grenade']['quantity'] = 2
    attacker_actions['elixir']['quantity'] = 1
    
    # Support
    support_actions['potion']['quantity'] = 3
    support_actions['grenade']['quantity'] = 2
    support_actions['elixir']['quantity'] = 1
    
    print("Quantities reset for both players")


def calculate_scores_attacker(hp, mp, hp_enemy):
    """Score per agente attacker - priorità danno"""
    hp_max = 3260
    mp_max = 132
    hp_enemy_max = 4000
    
    score_dict = {}
    
    for nome, a in attacker_actions.items():
        # Action not executable
        if a['quantity'] == 0 or mp < a['mp_cost']:
            score_dict[nome] = 0
            continue
        
        # Calcoli base
        danno_effettivo = min(a.get('damage', 0), hp_enemy)
        cura_effettiva = min(a.get('heal', 0), hp_max - hp)
        recupero_mp = a.get('mp_heal', 0)
        
        # Pesi dinamici - ATTACKER FOCUSED
        p_danno = 2.5 + (1 - hp_enemy / hp_enemy_max)  # ⬆️ Peso danno alto
        p_cura = 3.0 if hp < 0.2 * hp_max else (1.0 if hp < 0.5 * hp_max else 0.1)  # Cure solo se critico
        p_mp = 2.0 if mp < 25 else 0.2
        p_costo_mp = 0.3
        
        # Score
        efficacia = (
            p_danno * danno_effettivo +
            p_cura * cura_effettiva +
            p_mp * recupero_mp -
            p_costo_mp * a['mp_cost']
        )
        score_dict[nome] = max(efficacia, 0)
    
    # Normalizzazione
    max_score = max(score_dict.values()) if score_dict.values() else 1
    if max_score > 0:
        score_dict = {k: v / max_score for k, v in score_dict.items()}
    
    return score_dict


def calculate_scores_support(hp_self, hp_mate, mp, hp_enemy):
    """Score per agente support - priorità cure e supporto"""
    hp_max = 3260
    mp_max = 132
    hp_enemy_max = 4000
    
    score_dict = {}
    
    for nome, a in support_actions.items():
        # Action not executable
        if a['quantity'] == 0 or mp < a['mp_cost']:
            score_dict[nome] = 0
            continue
        
        # Calcoli base
        danno_effettivo = min(a.get('damage', 0), hp_enemy)
        heal_value = a.get('heal', 0)
        recupero_mp = a.get('mp_heal', 0)
        
        # Determina tipo di cura basandosi sul nome
        if nome in ['cura spell']:  # Auto-cure
            cura_self = min(heal_value, hp_max - hp_self)
            cura_mate = 0
        elif nome in ['cura_m', 'cura_totm']:  # Cura mate
            cura_self = 0
            cura_mate = min(heal_value, hp_max - hp_mate)
        elif nome in ['cura_tot', 'splash']:  # Cura entrambi
            cura_self = min(heal_value, hp_max - hp_self)
            cura_mate = min(heal_value, hp_max - hp_mate)
        else:  # Altri items/actions
            cura_self = min(heal_value, hp_max - hp_self) if heal_value > 0 else 0
            cura_mate = 0
        
        # Pesi dinamici - SUPPORT FOCUSED
        p_danno = 0.5
        
        # Priorità cure
        if hp_mate < 0.3 * hp_max:  # Mate critico
            p_cura_mate = 4.0
            p_cura_self = 1.0
        elif hp_self < 0.3 * hp_max:  # Self critico
            p_cura_mate = 1.5
            p_cura_self = 3.0
        elif hp_mate < 0.6 * hp_max:  # Mate medio
            p_cura_mate = 2.5
            p_cura_self = 1.0 if hp_self < 0.6 * hp_max else 0.2
        else:  # Tutti sani
            p_cura_mate = 0.5
            p_cura_self = 0.5
        
        p_mp = 2.5 if mp < 40 else 0.3
        p_costo_mp = 0.2
        
        # Score
        efficacia = (
            p_danno * danno_effettivo +
            p_cura_self * cura_self +
            p_cura_mate * cura_mate +
            p_mp * recupero_mp -
            p_costo_mp * a['mp_cost']
        )
        score_dict[nome] = max(efficacia, 0)
    
    # Normalizzazione
    max_score = max(score_dict.values()) if score_dict.values() else 1
    if max_score > 0:
        score_dict = {k: v / max_score for k, v in score_dict.items()}
    
    return score_dict
