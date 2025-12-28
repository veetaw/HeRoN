import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
import os
import pickle
import random
import numpy as np

# ==================== CONFIGURAZIONE GPU OTTIMIZZATA ====================
# Configura TensorFlow per utilizzare 2× NVIDIA RTX 5000 ADA (48GB VRAM ciascuna)

# 1. Abilita memoria growth per evitare OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU rilevate: {len(gpus)}× {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️  Errore configurazione GPU: {e}")
else:
    print("⚠️  Nessuna GPU rilevata - usando CPU")

# 2. Abilita Mixed Precision (FP16) per 2x velocità su RTX 5000 ADA
# TEMPORANEAMENTE DISABILITATO per WSL2 - decommentare quando CUDA è configurato
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print(f"🔥 Mixed Precision abilitata: {policy.name}")
print("⚠️  Mixed Precision disabilitata (WSL2 compatibility mode)")

# 3. Abilita XLA (Accelerated Linear Algebra) per ottimizzare Xeon + GPU
# TEMPORANEAMENTE DISABILITATO per WSL2 - decommentare quando CUDA è configurato
# tf.config.optimizer.set_jit(True)
# print("⚡ XLA compilation abilitata")
print("⚠️  XLA disabilitata (WSL2 compatibility mode)")

# 4. Configura Multi-GPU Strategy per 2× RTX 5000 ADA
strategy = tf.distribute.MirroredStrategy()
print(f"🎯 Multi-GPU Strategy: {strategy.num_replicas_in_sync} GPU attive")
# =========================================================================


class DQNAgent:
    def __init__(self, state_size, action_size, load_model_path):
        self.state_size = state_size
        self.action_size = action_size
        
        # OTTIMIZZAZIONE: NumPy circular buffer invece di deque per O(1) performance
        self.max_memory = 6000
        self.memory = np.empty(self.max_memory, dtype=object)
        self.memory_index = 0
        self.memory_size = 0
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9975  # Rallentato per mantenere exploration più a lungo
        self.learning_rate = 0.0015  # Aumentato per dimenticare più velocemente cattive esperienze
        
        if load_model_path:
            self.load(load_model_path)
            self.epsilon = 0.1
        else:
            self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        #model = Sequential()
        #model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(24, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def load(self, path_prefix):
        self.model = load_model(f"{path_prefix}.keras")
        memory_path = f"{path_prefix}_memory.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                saved_data = pickle.load(f)
                # Gestisci sia vecchio formato (deque) che nuovo (circular buffer)
                if isinstance(saved_data, dict):
                    self.memory = saved_data['buffer']
                    self.memory_index = saved_data['index']
                    self.memory_size = saved_data['size']
                else:
                    # Converti vecchio formato deque a circular buffer
                    old_memory = list(saved_data)
                    self.memory_size = min(len(old_memory), self.max_memory)
                    for i, item in enumerate(old_memory[-self.memory_size:]):
                        self.memory[i] = item
                    self.memory_index = self.memory_size % self.max_memory

    def remember(self, state, action, reward, next_state, done, valid_actions=None):
        """Salva l'esperienza nella memoria usando circular buffer - O(1)"""
        self.memory[self.memory_index] = (state, action, reward, next_state, done, valid_actions)
        self.memory_index = (self.memory_index + 1) % self.max_memory
        self.memory_size = min(self.memory_size + 1, self.max_memory)

    # TODO: player_index va tolto dato che ora ci sono due agenti
    def act(self, state, env, player_index):
        valid_actions = env.get_valid_actions(player_index)
        
        if len(valid_actions) == 0:
            return None
        
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        act_values = self.model.predict(state, verbose=0)
        masked_q_values = np.full(act_values.shape[1], -np.inf)
        masked_q_values[valid_actions] = act_values[0][valid_actions]
        
        return np.argmax(masked_q_values)


    # OTTIMIZZATO: Batch predict/fit + NumPy circular buffer per massime prestazioni
    def replay(self, batch_size, env, player_index):
        """Versione ottimizzata con batch predict/fit e circular buffer - O(1) per tutto"""
        # NumPy random choice su circular buffer - molto più veloce di random.sample su deque
        indices = np.random.choice(self.memory_size, batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]
        
        # Estrai batch di stati (invece di processarli uno alla volta)
        states = np.vstack([exp[0] for exp in minibatch])
        next_states = np.vstack([exp[3] for exp in minibatch])
        
        # BATCH PREDICT - 2 chiamate invece di batch_size*2 chiamate
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        # Prepara tutti i target in una volta
        targets = current_q_values.copy()
        
        for i, experience in enumerate(minibatch):
            # Gestisci entrambi i formati (con/senza valid_actions)
            if len(experience) == 6:
                state, action, reward, next_state, done, valid_actions = experience
            else:
                state, action, reward, next_state, done = experience
                valid_actions = None
            
            target = reward
            
            if not done:
                q_values_next = next_q_values[i]
                
                if valid_actions is not None and len(valid_actions) > 0:
                    # Usa le valid_actions salvate
                    masked_q = np.full_like(q_values_next, -np.inf)
                    masked_q[valid_actions] = q_values_next[valid_actions]
                    max_q = np.max(masked_q)
                else:
                    # Fallback: usa tutti i Q-values
                    max_q = np.max(q_values_next)
                
                target = reward + self.gamma * max_q
            
            targets[i][action] = target
        
        # BATCH FIT - 1 chiamata invece di batch_size chiamate
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path_prefix):
        self.model.save(f"{path_prefix}.keras")
        # Salva circular buffer con metadata
        memory_data = {
            'buffer': self.memory,
            'index': self.memory_index,
            'size': self.memory_size
        }
        with open(f"{path_prefix}_memory.pkl", 'wb') as f:
            pickle.dump(memory_data, f)
        with open(f"{path_prefix}_epsilon.txt", 'w') as f:
            f.write(str(self.epsilon))

'''
    def replay(self, batch_size, env):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                valid_actions = env.get_valid_actions()
                q_values_next = self.model.predict(next_state, verbose=0)[0]
                filtered_q = [q_values_next[i] for i in valid_actions]
                target = reward + self.gamma * max(filtered_q) if filtered_q else reward

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
'''


