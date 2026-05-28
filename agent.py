import gymnasium as gym
import itertools
import random
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import yaml
import os

from dqn import DQN
from experience_replay import ReplayMemory

import flappy_bird_gymnasium


DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"

os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent():

    def __init__(self, hyperparameter_set):

        with open('hyperparameters.yml', 'r') as file:

            all_hyperparameter_sets = yaml.safe_load(file)

            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters
        self.env_id = hyperparameters['env_id']

        self.learning_rate_a = hyperparameters['learning_rate_a']

        self.discount_factor_g = hyperparameters['discount_factor_g']

        self.network_sync_rate = hyperparameters['network_sync_rate']

        self.replay_memory_size = hyperparameters['replay_memory_size']

        self.mini_batch_size = hyperparameters['mini_batch_size']

        self.epsilon_init = hyperparameters['epsilon_init']

        self.epsilon_decay = hyperparameters['epsilon_decay']

        self.epsilon_min = hyperparameters['epsilon_min']

        self.stop_on_reward = hyperparameters['stop_on_reward']

        self.fc1_nodes = hyperparameters['fc1_nodes']

        self.env_make_params = hyperparameters.get(
            'env_make_params',
            {}
        )

        # Neural Network
        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = None

        # File Paths
        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f'{self.hyperparameter_set}.log'
        )

        self.CARTPOLE_LOG_FILE = os.path.join(
            RUNS_DIR,
            'cartpole.log'
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f'{self.hyperparameter_set}.pt'
        )

        self.GRAPH_FILE = os.path.join(
            RUNS_DIR,
            f'{self.hyperparameter_set}.png'
        )

        self.CARTPOLE_GRAPH_FILE = os.path.join(
            RUNS_DIR,
            'cartpole.png'
        )

    def run(self, is_training=True, render=False):

        if is_training:

            start_time = datetime.now()

            last_graph_update_time = start_time

            log_message = (
                f"{start_time.strftime(DATE_FORMAT)}:"
                " Training starting..."
            )

            print(log_message)

            with open(self.LOG_FILE, 'w') as file:

                file.write(log_message + '\n')

            with open(self.CARTPOLE_LOG_FILE, 'w') as file:

                file.write(log_message + '\n')

        # Create Environment
        env = gym.make(
            self.env_id,
            render_mode='human' if render else None,
            **self.env_make_params
        )

        num_actions = env.action_space.n

        num_states = env.observation_space.shape[0]

        rewards_per_episode = []

        # Policy Network
        policy_dqn = DQN(
            num_states,
            num_actions,
            self.fc1_nodes
        ).to(device)

        if is_training:

            epsilon = self.epsilon_init

            # Replay Memory
            memory = ReplayMemory(
                self.replay_memory_size
            )

            # Target Network
            target_dqn = DQN(
                num_states,
                num_actions,
                self.fc1_nodes
            ).to(device)

            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )

            # Optimizer
            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(),
                lr=self.learning_rate_a
            )

            epsilon_history = []

            step_count = 0

            best_reward = -9999999

        else:

            # Load trained model
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE)
            )

            policy_dqn.eval()

        # Infinite Episodes
        for episode in itertools.count():

            state, _ = env.reset()

            state = torch.tensor(
                state,
                dtype=torch.float,
                device=device
            )

            terminated = False

            episode_reward = 0.0

            # Episode Loop
            while (
                not terminated and
                episode_reward < self.stop_on_reward
            ):

                # Epsilon-Greedy
                if (
                    is_training and
                    random.random() < epsilon
                ):

                    action = env.action_space.sample()

                    action = torch.tensor(
                        action,
                        dtype=torch.int64,
                        device=device
                    )

                else:

                    with torch.no_grad():

                        action = policy_dqn(
                            state.unsqueeze(dim=0)
                        ).squeeze().argmax()

                # Execute Action
                new_state, reward, terminated, truncated, info = env.step(
                    action.item()
                )

                # ===== REWARD SHAPING: Add bonus for surviving =====
                if not terminated:
                    reward = reward + 0.1
                # ===== END REWARD SHAPING =====

                episode_reward += reward

                # Convert to tensors
                new_state = torch.tensor(
                    new_state,
                    dtype=torch.float,
                    device=device
                )

                reward = torch.tensor(
                    reward,
                    dtype=torch.float,
                    device=device
                )

                if is_training:

                    # Save transition
                    memory.append(
                        (
                            state,
                            action,
                            new_state,
                            reward,
                            terminated
                        )
                    )

                    step_count += 1

                state = new_state

            rewards_per_episode.append(
                episode_reward
            )

            print(
                f"Episode {episode} | Reward = {episode_reward}"
            )

            if is_training:

                # Save Best Model
                if episode_reward > best_reward:

                    log_message = (
                        f"{datetime.now().strftime(DATE_FORMAT)}:"
                        f" New best reward "
                        f"{episode_reward:0.1f}"
                    )

                    print(log_message)

                    with open(self.LOG_FILE, 'a') as file:

                        file.write(log_message + '\n')

                    with open(self.CARTPOLE_LOG_FILE, 'a') as file:

                        file.write(log_message + '\n')

                    torch.save(
                        policy_dqn.state_dict(),
                        self.MODEL_FILE
                    )

                    best_reward = episode_reward

                # Update Graph
                current_time = datetime.now()

                if current_time - last_graph_update_time > timedelta(seconds=10):

                    self.save_graph(
                        rewards_per_episode,
                        epsilon_history
                    )

                    last_graph_update_time = current_time

                # Train Only If Enough Experience
                if len(memory) > self.mini_batch_size:

                    mini_batch = memory.sample(
                        self.mini_batch_size
                    )

                    self.optimize(
                        mini_batch,
                        policy_dqn,
                        target_dqn
                    )

                    epsilon = max(
                        epsilon * self.epsilon_decay,
                        self.epsilon_min
                    )

                    epsilon_history.append(epsilon)

                    # Sync Networks
                    if step_count > self.network_sync_rate:

                        target_dqn.load_state_dict(
                            policy_dqn.state_dict()
                        )

                        step_count = 0

    def save_graph(
        self,
        rewards_per_episode,
        epsilon_history
    ):

        fig = plt.figure(1)

        mean_rewards = np.zeros(
            len(rewards_per_episode)
        )

        for x in range(len(mean_rewards)):

            mean_rewards[x] = np.mean(
                rewards_per_episode[
                    max(0, x-99):(x+1)
                ]
            )

        # Rewards Plot
        plt.subplot(121)

        plt.xlabel('Episodes')

        plt.ylabel('Mean Rewards')

        plt.plot(mean_rewards)

        # Epsilon Plot
        plt.subplot(122)

        plt.xlabel('Time Steps')

        plt.ylabel('Epsilon Decay')

        plt.plot(epsilon_history)

        plt.subplots_adjust(
            wspace=1.0,
            hspace=1.0
        )

        fig.savefig(self.GRAPH_FILE)

        fig.savefig(self.CARTPOLE_GRAPH_FILE)

        plt.close(fig)

    def optimize(
        self,
        mini_batch,
        policy_dqn,
        target_dqn
    ):

        states, actions, new_states, rewards, terminations = zip(
            *mini_batch
        )

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)

        terminations = torch.tensor(
            terminations
        ).float().to(device)

        # Compute Target Q Values
        with torch.no_grad():

            target_q = rewards + (
                1 - terminations
            ) * self.discount_factor_g * target_dqn(
                new_states
            ).max(dim=1)[0]

        # Current Q Values
        current_q = policy_dqn(
            states
        ).gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze()

        # Loss
        loss = self.loss_fn(
            current_q,
            target_q
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train or test model.'
    )

    parser.add_argument(
        'hyperparameters',
        help=''
    )

    parser.add_argument(
        '--train',
        help='Training mode',
        action='store_true'
    )

    args = parser.parse_args()

    dql = Agent(
        hyperparameter_set=args.hyperparameters
    )

    dql.run(
        is_training=args.train,
        render=True
    )