# -*- coding: utf-8 -*-
"""
Control of Inverted Pendulum with Actor-Critic (REINFORCE with Baseline)

This script implements an Actor-Critic algorithm to solve the Pendulum-v0
environment. This is an improvement over vanilla REINFORCE.

This version focuses on:
- An Actor network (policy) and a Critic network (value function).
- A training loop that uses the Advantage function for stable updates.
- Saving and loading the trained policy (Actor) network.
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
# Use CUDA for GPU acceleration if available, otherwise use CPU.
# This ensures the code runs on different hardware setups.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# 1. Actor-Critic Network and Helper Class
# =============================================================================

class PolicyOpt(object):
    """
    A base class to define and manage the Actor-Critic networks and related
    functionality like action computation, saving, and loading.
    """

    def __init__(self, env, linear=False, stochastic=True, hidden_size=32):
        """
        Initializes the policy (Actor) and value (Critic) networks.

        Args:
            env (gym.Env): The environment the agent will interact with.
            linear (bool): If True, use a simple linear layer for the policy.
            stochastic (bool): If True, the policy will sample actions from a distribution.
            hidden_size (int): The number of neurons in the hidden layers of the networks.
        """
        self.env = env
        self.stochastic = stochastic
        # Get the dimensions of the observation (state) and action spaces from the environment.
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))

        # Get the action space boundaries and convert them to PyTorch tensors.
        self.act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=DEVICE)
        self.act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=DEVICE)

        # --- Actor Network (The "Policy") ---
        # This network learns which action to take in a given state.
        actor_layers = []
        last = self.obs_dim
        if linear:
            actor_layers.append(nn.Linear(last, self.act_dim))
        else:
            # A simple neural network: Linear -> Tanh activation -> Linear
            actor_layers += [nn.Linear(last, hidden_size), nn.Tanh(), nn.Linear(hidden_size, self.act_dim)]
        # Create the sequential model and move it to the selected device (CPU or GPU).
        self.policy_net = nn.Sequential(*actor_layers).to(DEVICE)

        # --- Critic Network (The "Value Function") ---
        # This network learns to estimate the expected return (value) of a given state.
        self.value_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Outputs a single value: the estimated return.
        ).to(DEVICE)

        # The standard deviation of the policy's action distribution is a learnable parameter.
        # We learn the log of the std dev for numerical stability.
        self.log_std = nn.Parameter(torch.full((self.act_dim,), -0.5, device=DEVICE))

        # Optimizer will be defined in the training class.
        self.optimizer = None
        self.gamma = 0.99  # Default discount factor for future rewards.

    def _dist(self, obs_tensor: torch.Tensor) -> Normal:
        """
        Takes an observation tensor and returns a Normal (Gaussian) distribution
        over the actions, as defined by the policy network.

        Args:
            obs_tensor (torch.Tensor): The state/observation.

        Returns:
            torch.distributions.Normal: A distribution from which to sample an action.
        """
        # The actor network outputs the mean of the action distribution.
        mu = self.policy_net(obs_tensor)
        # The standard deviation is derived from the learnable log_std parameter.
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def compute_action(self, s, stochastic=None):
        """
        Computes an action for a given state using the Actor network.

        Args:
            s (np.ndarray): The current state observation.
            stochastic (bool, optional): If True, sample from the distribution.
                                         If False, take the mean (deterministic action).

        Returns:
            np.ndarray: The computed action.
        """
        use_stochastic = self.stochastic if stochastic is None else stochastic
        # Convert the numpy state to a PyTorch tensor and add a batch dimension if needed.
        s_t = torch.as_tensor(np.array(s, dtype=np.float32), device=DEVICE)
        if s_t.ndim == 1:
            s_t = s_t.unsqueeze(0)

        # We don't need to track gradients when just computing an action.
        with torch.no_grad():
            # Get the action distribution for the current state.
            dist = self._dist(s_t)
            # Sample for exploration during training, or take the mean for evaluation.
            a = dist.sample() if use_stochastic else dist.mean

        # Ensure the action is within the environment's valid bounds.
        a = a.clamp(self.act_low, self.act_high)
        # Convert the action tensor back to a numpy array for the gym environment.
        return a.detach().cpu().numpy()

    def save_policy(self, filepath):
        """Saves the policy (Actor) network's state dictionary to a file."""
        print(f"\nSaving policy to {filepath}...")
        torch.save(self.policy_net.state_dict(), filepath)
        print("Policy saved successfully.")

    def load_policy(self, filepath):
        """Loads policy (Actor) network's weights from a file."""
        if not os.path.exists(filepath):
            print(f"Error: No file found at {filepath}")
            return
        print(f"\nLoading policy from {filepath}...")
        # Load the saved weights into the policy_net model.
        self.policy_net.load_state_dict(torch.load(filepath, map_location=DEVICE))
        print("Policy loaded successfully.")


# =============================================================================
# 2. Actor-Critic Algorithm Implementation
# =============================================================================

class ActorCritic(PolicyOpt):
    """
    Implements the Actor-Critic (REINFORCE with Baseline) algorithm by inheriting
    from PolicyOpt and defining the training logic.
    """

    def train(
            self,
            num_iterations: int = 100,
            steps_per_iteration: int = 5000,
            learning_rate: float = 1e-3,
            gamma: float = 0.95,
    ):
        """
        The main training loop for the Actor-Critic algorithm.
        """
        self.gamma = gamma
        # The Adam optimizer will update the parameters of BOTH the Actor and the Critic networks,
        # as well as the learnable log_std for the action distribution.
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()) + [self.log_std],
            lr=float(learning_rate),
        )

        episode_returns = []
        total_steps = 0

        # Main training loop. Each iteration is one policy update.
        for it in range(num_iterations):
            # Buffers to store data collected in this iteration.
            batch_states, batch_actions, batch_returns = [], [], []
            steps_in_batch = 0

            # --- Data Collection Phase ---
            # Collect a batch of experience by running episodes until we have enough steps.
            while steps_in_batch < steps_per_iteration:
                obs = self.env.reset()
                ep_states, ep_actions, ep_rewards = [], [], []
                done = False

                # Run a single episode until it terminates.
                while not done:
                    # Get a stochastic action from the current policy.
                    a_np = self.compute_action(obs, stochastic=True)[0]
                    # Take a step in the environment.
                    obs_next, r, done, info = self.env.step(a_np)

                    # Store the experience tuple.
                    ep_states.append(obs)
                    ep_actions.append(a_np)
                    ep_rewards.append(float(r))

                    obs = obs_next
                    steps_in_batch += 1

                # --- Reward-to-Go Calculation ---
                # After an episode finishes, calculate the discounted return for each step.
                G = np.zeros_like(ep_rewards, dtype=np.float32)
                g = 0.0
                # Iterate backwards through the episode's rewards.
                for t in range(len(ep_rewards) - 1, -1, -1):
                    g = ep_rewards[t] + gamma * g
                    G[t] = g

                # Add the episode's data to the batch buffers.
                batch_states.extend(ep_states)
                batch_actions.extend(ep_actions)
                batch_returns.extend(G.tolist())
                episode_returns.append(sum(ep_rewards))

            # --- Learning Phase ---
            # Convert collected data into PyTorch tensors.
            s_b = torch.as_tensor(np.array(batch_states, dtype=np.float32), device=DEVICE)
            a_b = torch.as_tensor(np.array(batch_actions, dtype=np.float32), device=DEVICE)
            G_b = torch.as_tensor(np.array(batch_returns, dtype=np.float32), device=DEVICE)

            # --- CRITIC'S JOB: Predict state values ---
            # Get the value estimate for each state in the batch from the critic network.
            V_b = self.value_net(s_b).squeeze()

            # --- ADVANTAGE CALCULATION ---
            # Advantage = Actual Returns (G_b) - Predicted Returns (V_b)
            # This tells us how much better or worse the actual outcome was than expected.
            # .detach() is used so that gradients from the actor loss don't flow back into the critic.
            advantage_b = (G_b - V_b).detach()

            # Normalize the advantage to have zero mean and unit variance. This stabilizes training.
            normalized_advantage_b = (advantage_b - advantage_b.mean()) / (advantage_b.std() + 1e-8)

            # --- ACTOR LOSS ---
            # The actor's goal is to take actions that lead to high advantages.
            dist_b = self._dist(s_b)
            logp = dist_b.log_prob(a_b).sum(dim=-1)
            # We multiply the log-probability of each action by its normalized advantage.
            # The negative sign is because optimizers perform minimization.
            actor_loss = -(logp * normalized_advantage_b).mean()

            # --- CRITIC LOSS ---
            # The critic's goal is to make its value predictions (V_b) as close as possible
            # to the actual returns (G_b). Mean Squared Error is a standard loss for this.
            critic_loss = nn.MSELoss()(V_b, G_b)

            # --- COMBINED LOSS AND UPDATE ---
            # We combine the actor and critic losses into a single loss function.
            loss = actor_loss + critic_loss

            # Perform the backpropagation and optimization step.
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_steps += steps_in_batch
            # Print progress periodically.
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
    Loads training results from a CSV file and plots the learning curve.
    """
    print("\nPlotting training results...")
    try:
        # Load the saved episode returns.
        all_results = np.genfromtxt(results_file, delimiter=",")
        if all_results.ndim == 1:
            all_results = np.expand_dims(all_results, axis=0)

        # Helper function to smooth the noisy reward curve.
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size), 'valid') / window_size

        window = 20  # Smoothing window size.

        # Apply smoothing to each trial's results.
        smoothed_results = [moving_average(res, window) for res in all_results]
        smoothed_results = np.array(smoothed_results)

        # Calculate the mean and standard deviation across trials.
        mean_rewards = np.mean(smoothed_results, axis=0)
        std_rewards = np.std(smoothed_results, axis=0)
        x_vals = np.arange(len(mean_rewards))

        # Create the plot.
        plt.figure(figsize=(12, 7))
        plt.plot(x_vals, mean_rewards, label="Mean Episode Reward (Actor-Critic)", color='blue')
        # Show the standard deviation as a shaded area.
        plt.fill_between(
            x_vals,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            color='blue',
            alpha=0.2,
            label="Standard Deviation"
        )
        plt.title("Actor-Critic Training on Pendulum-v0")
        plt.xlabel(f"Episodes (Smoothed over {window} episodes)")
        plt.ylabel("Total Episode Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

    except IOError:
        print(f"Error: Results file not found at '{results_file}'. Please run training first.")


def evaluate_policy(env, policy_agent, episodes=5):
    """
    Runs several episodes with the trained (deterministic) policy to gauge performance.
    """
    print("\nEvaluating trained policy...")
    total_rewards = []
    for i in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Use the deterministic action (mean of the distribution) for evaluation.
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
    Main function to orchestrate the training, saving, loading, and evaluation process.
    """
    # --- Configuration ---
    ENV_NAME = "Pendulum-v0"
    NUM_TRIALS = 1  # Set to >1 for more robust plots with standard deviation.
    POLICY_FILE = "trained_pendulum_policy.pth"
    RESULTS_FILE = "InvertedPendulum_results.csv"

    # --- Train the agent ---
    train_env = gym.make(ENV_NAME)
    all_trial_rewards = []
    trained_alg = None

    for i in range(NUM_TRIALS):
        print(f"\n==== Training Run {i + 1}/{NUM_TRIALS} ====")
        alg = ActorCritic(train_env, stochastic=True)
        res = alg.train(
            learning_rate=0.002,
            gamma=0.95,
            num_iterations=200,
            steps_per_iteration=5000
        )
        all_trial_rewards.append(np.array(res))
        if i == NUM_TRIALS - 1:
            trained_alg = alg  # Keep the agent from the last trial.

    # Save episode returns to a CSV for plotting.
    np.savetxt(RESULTS_FILE, np.array(all_trial_rewards), delimiter=",")
    train_env.close()

    # --- Save the trained policy's weights ---
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
        # Create a new agent instance.
        loaded_agent = ActorCritic(eval_env, stochastic=True)
        # Load the saved weights into the new agent's policy network.
        loaded_agent.load_policy(POLICY_FILE)

        # Evaluate the performance of the loaded agent.
        evaluate_policy(eval_env, loaded_agent)
        eval_env.close()


# This ensures the main() function is called only when the script is executed directly.
if __name__ == "__main__":
    main()
