# -*- coding: utf-8 -*-
"""
Control of Inverted Pendulum with REINFORCE Algorithm (Save/Load Version)

This script implements the REINFORCE algorithm to solve the Pendulum-v0
environment. It has been modified to remove all rendering dependencies.

This version focuses on:
- A PyTorch-based policy network.
- The REINFORCE training loop with return normalization for stability.
- Saving the trained policy's weights to a file.
- Loading the saved weights into a new policy object.
- Plotting the training results.

Requirements:
- Python >= 3.6
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gym
import matplotlib.pyplot as plt
import os

# --- Global Settings ---
# Use CUDA if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# 1. Policy Network and Helper Class
# =============================================================================

class PolicyOpt(object):
    """
    A class to define and manage the policy for the REINFORCE algorithm.
    """

    def __init__(self, env, linear=False, stochastic=True, hidden_size=32):
        """
        Initializes the policy network and its parameters.
        """
        self.env = env
        self.stochastic = stochastic
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))

        # Action bounds (on device)
        self.act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=DEVICE)
        self.act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=DEVICE)

        # Build policy network
        layers = []
        last = self.obs_dim
        if linear:
            layers.append(nn.Linear(last, self.act_dim))
        else:
            layers += [nn.Linear(last, hidden_size), nn.Tanh(), nn.Linear(hidden_size, self.act_dim)]
        self.policy_net = nn.Sequential(*layers).to(DEVICE)

        # Log-std as a free parameter (state-independent)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), -0.5, device=DEVICE))

        # Optimizer placeholder (initialized in REINFORCE.train)
        self.optimizer = None
        self.gamma = 0.99  # Default discount factor

    def _dist(self, obs_tensor: torch.Tensor) -> Normal:
        """Creates a Normal distribution from the policy network output."""
        mu = self.policy_net(obs_tensor)
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def compute_action(self, s, stochastic=None):
        """
        Computes an action for a given state.
        """
        use_stochastic = self.stochastic if stochastic is None else stochastic
        s_t = torch.as_tensor(np.array(s, dtype=np.float32), device=DEVICE)
        if s_t.ndim == 1:
            s_t = s_t.unsqueeze(0)
        with torch.no_grad():
            dist = self._dist(s_t)
            a = dist.sample() if use_stochastic else dist.mean
        a = a.clamp(self.act_low, self.act_high)
        return a.detach().cpu().numpy()

    def save_policy(self, filepath):
        """Saves the policy network's weights to a file."""
        print(f"\nSaving policy to {filepath}...")
        torch.save(self.policy_net.state_dict(), filepath)
        print("Policy saved successfully.")

    def load_policy(self, filepath):
        """Loads policy network's weights from a file."""
        if not os.path.exists(filepath):
            print(f"Error: No file found at {filepath}")
            return
        print(f"\nLoading policy from {filepath}...")
        self.policy_net.load_state_dict(torch.load(filepath, map_location=DEVICE))
        print("Policy loaded successfully.")


# =============================================================================
# 2. REINFORCE Algorithm Implementation
# =============================================================================

class REINFORCE(PolicyOpt):
    """
    Implements the REINFORCE (Monte-Carlo Policy Gradient) algorithm.
    """

    def train(
            self,
            num_iterations: int = 100,
            steps_per_iteration: int = 5000,
            learning_rate: float = 1e-3,
            gamma: float = 0.95,
    ):
        """
        The main training loop for the REINFORCE algorithm.
        """
        self.gamma = gamma
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + [self.log_std],
            lr=float(learning_rate),
        )

        episode_returns = []
        total_steps = 0

        for it in range(num_iterations):
            batch_states, batch_actions, batch_returns = [], [], []
            steps_in_batch = 0

            while steps_in_batch < steps_per_iteration:
                obs = self.env.reset()
                ep_states, ep_actions, ep_rewards = [], [], []
                done = False

                while not done:
                    a_np = self.compute_action(obs, stochastic=True)[0]
                    obs_next, r, done, info = self.env.step(a_np)

                    ep_states.append(obs)
                    ep_actions.append(a_np)
                    ep_rewards.append(float(r))

                    obs = obs_next
                    steps_in_batch += 1

                # Compute reward-to-go for the completed episode
                G = np.zeros_like(ep_rewards, dtype=np.float32)
                g = 0.0
                for t in range(len(ep_rewards) - 1, -1, -1):
                    g = ep_rewards[t] + gamma * g
                    G[t] = g

                batch_states.extend(ep_states)
                batch_actions.extend(ep_actions)
                batch_returns.extend(G.tolist())
                episode_returns.append(sum(ep_rewards))

            # --- Perform the policy gradient update ---
            s_b = torch.as_tensor(np.array(batch_states, dtype=np.float32), device=DEVICE)
            a_b = torch.as_tensor(np.array(batch_actions, dtype=np.float32), device=DEVICE)
            G_b = torch.as_tensor(np.array(batch_returns, dtype=np.float32), device=DEVICE)

            # Normalize the returns (reward-to-go)
            normalized_G_b = (G_b - G_b.mean()) / (G_b.std() + 1e-8)

            dist_b = self._dist(s_b)
            logp = dist_b.log_prob(a_b).sum(dim=-1)

            # Update loss to use the normalized returns
            loss = -(logp * normalized_G_b).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_steps += steps_in_batch
            if (it + 1) % max(1, num_iterations // 10) == 0:
                print(
                    f"[Iter {it + 1}/{num_iterations}] "
                    f"Steps: {total_steps} | Loss: {loss.item():.4f} | "
                    f"EpRet(last): {episode_returns[-1]:.1f}"
                )

        return episode_returns


# =============================================================================
# 3. Plotting and Evaluation Functions
# =============================================================================

def plot_results(results_file="InvertedPendulum_results.csv"):
    """
    Loads training results from a CSV and plots them.
    """
    print("\nPlotting training results...")
    try:
        all_results = np.genfromtxt(results_file, delimiter=",")
        if all_results.ndim == 1:
            all_results = np.expand_dims(all_results, axis=0)

        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size), 'valid') / window_size

        window = 20  # Smoothing window

        smoothed_results = [moving_average(res, window) for res in all_results]
        smoothed_results = np.array(smoothed_results)

        mean_rewards = np.mean(smoothed_results, axis=0)
        std_rewards = np.std(smoothed_results, axis=0)
        x_vals = np.arange(len(mean_rewards))

        plt.figure(figsize=(12, 7))
        plt.plot(x_vals, mean_rewards, label="Mean Episode Reward (REINFORCE)", color='blue')
        plt.fill_between(
            x_vals,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            color='blue',
            alpha=0.2,
            label="Standard Deviation"
        )
        plt.title("REINFORCE Training on Pendulum-v0")
        plt.xlabel(f"Episodes (Smoothed over {window} episodes)")
        plt.ylabel("Total Episode Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

    except IOError:
        print(f"Error: Results file not found at '{results_file}'. Please run training first.")


def evaluate_policy(env, policy_agent, episodes=5):
    """
    Runs a few episodes with the trained agent to see its performance.
    """
    print("\nEvaluating trained policy...")
    total_rewards = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy_agent.compute_action(obs, stochastic=False)[0]
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {i + 1}: Total reward = {episode_reward:.2f}")
    print(f"Average evaluation reward: {np.mean(total_rewards):.2f}")


# =============================================================================
# 4. Main Execution Block
# =============================================================================

def main():
    """
    Main function to run the experiment.
    """
    ENV_NAME = "Pendulum-v0"
    NUM_TRIALS = 1  # Set to >1 for more robust plots
    POLICY_FILE = "trained_pendulum_policy.pth"
    RESULTS_FILE = "InvertedPendulum_results.csv"

    # --- Train the agent ---
    train_env = gym.make(ENV_NAME)
    all_trial_rewards = []
    trained_alg = None

    for i in range(NUM_TRIALS):
        print(f"\n==== Training Run {i + 1}/{NUM_TRIALS} ====")
        alg = REINFORCE(train_env, stochastic=True)
        res = alg.train(
            learning_rate=0.002,
            gamma=0.95,
            num_iterations=200,
            steps_per_iteration=5000
        )
        all_trial_rewards.append(np.array(res))
        if i == NUM_TRIALS - 1:
            trained_alg = alg  # Save the last trained agent

    # Save results to file
    np.savetxt(RESULTS_FILE, np.array(all_trial_rewards), delimiter=",")
    train_env.close()

    # --- Save the trained policy ---
    if trained_alg:
        trained_alg.save_policy(POLICY_FILE)
    else:
        print("Training was not completed, so no policy to save.")
        return

    # --- Plot training results ---
    plot_results(RESULTS_FILE)

    # --- Example of loading and evaluating the policy ---
    if os.path.exists(POLICY_FILE):
        eval_env = gym.make(ENV_NAME)
        loaded_agent = REINFORCE(eval_env, stochastic=True)
        loaded_agent.load_policy(POLICY_FILE)

        # Evaluate the loaded agent
        evaluate_policy(eval_env, loaded_agent)
        eval_env.close()


if __name__ == "__main__":
    main()
