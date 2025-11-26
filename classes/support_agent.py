from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
import pickle
import random
import numpy as np


class DQNSupportAgent:
    def __init__(self, state_size, action_size, load_model_path):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
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
        self.model = load_model(f"{path_prefix}.keras") # enter model path
        memory_path = f"{path_prefix}_memory.pkl" # enter model path
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)

    def remember(self, state, action, reward, next_state, done, valid_actions=None):
        """Salva l'esperienza nella memoria (ora include valid_actions)"""
        self.memory.append((state, action, reward, next_state, done, valid_actions))

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

    def replay(self, batch_size, env, player_index):
        """Versione migliorata che usa le valid_actions salvate"""
        import random
        import numpy as np
        
        minibatch = random.sample(self.memory, batch_size)
        
        for experience in minibatch:
            # Gestisci entrambi i formati (con/senza valid_actions)
            if len(experience) == 6:
                state, action, reward, next_state, done, valid_actions = experience
            else:
                state, action, reward, next_state, done = experience
                valid_actions = None
            
            target = reward
            
            if not done:
                q_values_next = self.model.predict(next_state, verbose=0)[0]
                
                if valid_actions is not None and len(valid_actions) > 0:
                    # Usa le valid_actions salvate
                    masked_q = np.full_like(q_values_next, -np.inf)
                    masked_q[valid_actions] = q_values_next[valid_actions]
                    max_q = np.max(masked_q)
                else:
                    # Fallback: usa tutti i Q-values
                    max_q = np.max(q_values_next)
                
                target = reward + self.gamma * max_q
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path_prefix):
        self.model.save(f"{path_prefix}.keras") # enter model path
        with open(f"{path_prefix}_memory.pkl", 'wb') as f: # enter model path
            pickle.dump(self.memory, f)
        with open(f"{path_prefix}_epsilon.txt", 'w') as f: # enter model path
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


