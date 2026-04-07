"""COLA: Conflict Objective Regularization in Latent Space for Multi-Objective RL.

This module implements COLA, a general-policy MORL algorithm that learns in a shared latent space
and mitigates optimization conflicts across preferences.

Key features:
- Objective-agnostic Latent Dynamics Model (OADM): builds a shared latent space capturing environment dynamics
- Conflict Objective Regularization (COR): regularizes value updates when optimization directions conflict

Paper: COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective
Regularization in Latent Space (NeurIPS 2025)
"""

import os
import random
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from morl_baselines.common.tensorboard_logger import log as tensorboard_log, Table

from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import layer_init, mlp, polyak_update
from morl_baselines.common.weights import equally_spaced_weights


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class ReplayMemory:
    """Replay memory for COLA with preference conditioning."""

    def __init__(self, capacity: int, obs_shape: tuple, action_shape: tuple, reward_dim: int):
        """Initialize the replay memory.

        Args:
            capacity: Maximum capacity of the replay buffer
            obs_shape: Shape of observations
            action_shape: Shape of actions
            reward_dim: Dimension of reward vector
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reward_dim = reward_dim

    def push(self, state, preference, action, reward, next_state, done):
        """Push a transition to the replay buffer.

        Args:
            state: Current observation
            preference: Weight/preference vector
            action: Action taken
            reward: Vector reward received
            next_state: Next observation
            done: Whether episode terminated
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            np.array(state).copy(),
            np.array(preference).copy(),
            np.array(action).copy(),
            np.array(reward).copy(),
            np.array(next_state).copy(),
            np.array(done).copy(),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, to_tensor: bool = True, device: str = None):
        """Sample a batch of transitions.

        Args:
            batch_size: Number of samples
            to_tensor: Whether to convert to PyTorch tensors
            device: Device to put tensors on

        Returns:
            Batch of (states, preferences, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, preference, action, reward, next_state, done = map(np.stack, zip(*batch))
        experience_tuples = (state, preference, action, reward, next_state, done)
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x, dtype=th.float32).to(device), experience_tuples))
        return state, preference, action, reward, next_state, done

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class QMemory:
    """Memory buffer for storing historical Q-networks for regularization."""

    def __init__(self, capacity: int):
        """Initialize Q memory.

        Args:
            capacity: Maximum number of Q-networks to store
        """
        self.capacity = capacity
        self.reset()

    def append(self, q_network_state_dict):
        """Append a Q-network state dict to memory.

        Args:
            q_network_state_dict: State dict of Q-network to store
        """
        if self.capacity > 0:
            self._append(q_network_state_dict)

    def _append(self, q_network_state_dict):
        if len(self.buffer) < self.capacity:
            self.buffer.append(q_network_state_dict)
        else:
            self.buffer[self._p] = q_network_state_dict
        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def reset(self):
        """Reset the Q memory."""
        self._n = 0
        self._p = 0
        self.full = False
        self.buffer = []

    def sample(self) -> List[dict]:
        """Return all stored Q-network state dicts.

        Returns:
            List of Q-network state dicts
        """
        return self.buffer

    def __len__(self):
        """Return number of stored Q-networks."""
        return self._n


class LatentEncoder(nn.Module):
    """Latent encoder for COLA.

    Encodes states into a latent space and predicts latent dynamics.
    """

    def __init__(
        self,
        use_avg: bool,
        state_dim: int,
        action_dim: int,
        reward_dim: int,
        latent_dim: int,
    ):
        """Initialize the latent encoder.

        Args:
            use_avg: Whether to use L1 normalization on latent features
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            reward_dim: Dimension of reward space
            latent_dim: Dimension of latent space
        """
        super(LatentEncoder, self).__init__()
        self.use_avg = use_avg
        self.latent_dim = latent_dim

        # State encoder: s -> z
        self.z_encoder = nn.Sequential(
            nn.Linear(state_dim + reward_dim, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim),
        )

        # Dynamics predictor: (z, a) -> z'_pred
        self.z_dynamic_pre = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim),
        )

    def get_latent_features(self, states_weights: th.Tensor) -> th.Tensor:
        """Encode state-preference pair to latent space.

        Args:
            states_weights: Concatenated state and preference tensor

        Returns:
            Latent representation
        """
        if self.use_avg:
            latent_features = self._avg_l1_norm(self.z_encoder(states_weights))
        else:
            latent_features = self.z_encoder(states_weights)
        return latent_features

    def get_dynamic(self, z: th.Tensor, a: th.Tensor) -> th.Tensor:
        """Predict next latent state given current latent and action.

        Args:
            z: Current latent representation
            a: Action taken

        Returns:
            Predicted next latent state
        """
        pre_next_latent_features = self.z_dynamic_pre(th.cat([z, a], dim=-1))
        return pre_next_latent_features

    @staticmethod
    def _avg_l1_norm(x: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        """L1 normalization with averaging.

        Args:
            x: Input tensor
            eps: Small constant for numerical stability

        Returns:
            Normalized tensor
        """
        return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class TwinnedQNetwork(nn.Module):
    """Twin Q-networks for COLA operating on latent space."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        reward_dim: int,
        hidden_units: List[int] = [256, 256],
        use_critic_preference: bool = True,
    ):
        """Initialize twin Q-networks.

        Args:
            latent_dim: Dimension of latent space
            action_dim: Dimension of action space
            reward_dim: Dimension of reward space (output dim)
            hidden_units: List of hidden layer sizes
            use_critic_preference: Whether to condition Q on preferences
        """
        super(TwinnedQNetwork, self).__init__()
        self.use_critic_preference = use_critic_preference

        # Input dim depends on configuration
        # Base: latent_dim + action_dim
        # With preference: + reward_dim
        input_dim = latent_dim + action_dim
        if use_critic_preference:
            input_dim += reward_dim

        self.Q1 = self._build_q_network(input_dim, reward_dim, hidden_units)
        self.Q2 = self._build_q_network(input_dim, reward_dim, hidden_units)

    def _build_q_network(self, input_dim: int, output_dim: int, hidden_units: List[int]) -> nn.Sequential:
        """Build a Q-network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (reward dim)
            hidden_units: Hidden layer sizes

        Returns:
            Q-network as nn.Sequential
        """
        modules = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            modules.append(nn.Linear(prev_dim, hidden_dim))
            modules.append(nn.ReLU())
            prev_dim = hidden_dim
        modules.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*modules)

    def forward(
        self,
        z: th.Tensor,
        actions: th.Tensor,
        preferences: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass through both Q-networks.

        Args:
            z: Latent state representation
            actions: Actions
            preferences: Preference vectors (optional if use_critic_preference=False)

        Returns:
            Tuple of (Q1_values, Q2_values)
        """
        if self.use_critic_preference and preferences is not None:
            x = th.cat([z, actions, preferences], dim=-1)
        else:
            x = th.cat([z, actions], dim=-1)

        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class GaussianPolicy(nn.Module):
    """Gaussian policy for COLA with latent space conditioning."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_space: gym.spaces.Box,
        hidden_units: List[int] = [128, 128],
    ):
        """Initialize Gaussian policy.

        Args:
            input_dim: Input dimension (latent or state+preference)
            action_dim: Dimension of action space
            action_space: Gym action space for scaling
            hidden_units: Hidden layer sizes
        """
        super(GaussianPolicy, self).__init__()
        self.action_dim = action_dim

        # Build policy network
        self.policy = self._build_policy(input_dim, action_dim * 2, hidden_units)

        # Action rescaling
        self.register_buffer(
            "action_scale",
            th.tensor((action_space.high - action_space.low) / 2.0, dtype=th.float32),
        )
        self.register_buffer(
            "action_bias",
            th.tensor((action_space.high + action_space.low) / 2.0, dtype=th.float32),
        )

    def _build_policy(self, input_dim: int, output_dim: int, hidden_units: List[int]) -> nn.Sequential:
        """Build policy network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (2 * action_dim for mean and log_std)
            hidden_units: Hidden layer sizes

        Returns:
            Policy network as nn.Sequential
        """
        modules = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            modules.append(nn.Linear(prev_dim, hidden_dim))
            modules.append(nn.ReLU())
            prev_dim = hidden_dim
        modules.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*modules)

    def forward(self, inputs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass to get mean and log_std.

        Args:
            inputs: Input tensor (latent or state+preference)

        Returns:
            Tuple of (mean, log_std)
        """
        x = inputs
        mean, log_std = th.chunk(self.policy(x), 2, dim=-1)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, inputs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample action from policy.

        Args:
            inputs: Input tensor

        Returns:
            Tuple of (action, log_prob, mean_action)
        """
        mean, log_std = self.forward(inputs)
        std = log_std.exp()

        # Sample with reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # Apply tanh and rescale
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Compute log prob with tanh correction
        log_prob = normal.log_prob(x_t).sum(dim=-1)
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON).sum(dim=-1)
        log_prob = log_prob.clamp(-1e3, 1e3)

        mean_action = th.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action

    def get_action(self, inputs: th.Tensor) -> th.Tensor:
        """Get deterministic action (mean).

        Args:
            inputs: Input tensor

        Returns:
            Mean action
        """
        mean, _ = self.forward(inputs)
        return th.tanh(mean) * self.action_scale + self.action_bias


class COLA(MOAgent, MOPolicy):
    """COLA: Conflict Objective Regularization in Latent Space.

    A general-policy MORL algorithm that learns in a shared latent space and mitigates
    optimization conflicts across preferences.

    Paper: COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict
    Objective Regularization in Latent Space (NeurIPS 2025)

    Key features:
    - Objective-agnostic Latent Dynamics Model (OADM) for efficient knowledge sharing
    - Conflict Objective Regularization (COR) for stable value learning
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = int(1e6),
        q_memory_capacity: int = 10,
        batch_size: int = 256,
        latent_dim: int = 50,
        hidden_units: List[int] = [256, 256],
        policy_hidden_units: List[int] = [128, 128],
        alpha: float = 0.2,
        entropy_tuning: bool = True,
        learning_starts: int = 10000,
        gradient_updates: int = 1,
        target_update_interval: int = 1,
        encoder_update_freq: int = 1,
        old_q_update_freq: int = 1,
        regular_alpha: float = 0.1,
        regular_bar: float = 0.25,
        reward_coef: float = 1.0,
        dynamic_coef: float = 1.0,
        value_coef: float = 1.0,
        # Configuration flags
        use_critic_preference: bool = True,
        use_policy_preference: bool = True,
        policy_use_latent: bool = True,
        policy_use_s: bool = False,
        policy_use_w: bool = False,
        critic_use_s: bool = False,
        critic_use_a: bool = False,
        critic_use_both: bool = False,
        use_avg: bool = False,
        use_encoder_hardupdate: bool = False,
        # Logging
        project_name: str = "MORL-Baselines",
        experiment_name: str = "COLA",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """Initialize COLA agent.

        Args:
            env: Gym environment
            learning_rate: Learning rate for optimizers
            gamma: Discount factor
            tau: Soft update coefficient
            buffer_size: Replay buffer capacity
            q_memory_capacity: Q-memory capacity for regularization
            batch_size: Batch size for training
            latent_dim: Dimension of latent space
            hidden_units: Hidden units for Q-networks
            policy_hidden_units: Hidden units for policy network
            alpha: Initial entropy coefficient
            entropy_tuning: Whether to tune alpha automatically
            learning_starts: Steps before learning starts
            gradient_updates: Gradient updates per step
            target_update_interval: Interval for target network updates
            encoder_update_freq: Frequency of encoder target updates
            old_q_update_freq: Frequency of old Q-network updates
            regular_alpha: COR regularization strength
            regular_bar: COR stiffness threshold
            reward_coef: Coefficient for reward loss
            dynamic_coef: Coefficient for dynamics loss
            value_coef: Coefficient for value loss
            use_critic_preference: Condition Q on preferences
            use_policy_preference: Condition policy on preferences
            policy_use_latent: Use latent space for policy
            policy_use_s: Include state in policy input (with latent)
            policy_use_w: Include preference in policy input (with latent)
            critic_use_s: Include state in Q input
            critic_use_a: Include action in Q input
            critic_use_both: Use both current and next latent in Q
            use_avg: Use L1 normalization on latent features
            use_encoder_hardupdate: Use hard updates for encoder target
            project_name: Wandb project name
            experiment_name: Wandb experiment name
            wandb_entity: Wandb entity
            log: Whether to log to wandb
            seed: Random seed
            device: Device for training
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.policy_hidden_units = policy_hidden_units
        self.alpha = alpha
        self.entropy_tuning = entropy_tuning
        self.learning_starts = learning_starts
        self.gradient_updates = gradient_updates
        self.target_update_interval = target_update_interval
        self.encoder_update_freq = encoder_update_freq
        self.old_q_update_freq = old_q_update_freq
        self.regular_alpha = regular_alpha
        self.regular_bar = regular_bar
        self.reward_coef = reward_coef
        self.dynamic_coef = dynamic_coef
        self.value_coef = value_coef
        self.q_memory_capacity = q_memory_capacity

        # Configuration flags
        self.use_critic_preference = use_critic_preference
        self.use_policy_preference = use_policy_preference
        self.policy_use_latent = policy_use_latent
        self.policy_use_s = policy_use_s
        self.policy_use_w = policy_use_w
        self.critic_use_s = critic_use_s
        self.critic_use_a = critic_use_a
        self.critic_use_both = critic_use_both
        self.use_avg = use_avg
        self.use_encoder_hardupdate = use_encoder_hardupdate

        # Calculate policy input dimension
        if self.policy_use_latent:
            policy_input_dim = self.latent_dim
            if self.policy_use_s:
                policy_input_dim += self.observation_dim
            if self.policy_use_w:
                policy_input_dim += self.reward_dim
        else:
            policy_input_dim = self.observation_dim + self.reward_dim

        # Calculate critic input dimension
        if self.critic_use_both:
            critic_input_dim = self.latent_dim * 2
        else:
            critic_input_dim = self.latent_dim

        if self.critic_use_s:
            critic_input_dim += self.observation_dim
        if self.critic_use_a:
            critic_input_dim += self.action_dim

        # Initialize networks
        self.latent_encoder = LatentEncoder(
            self.use_avg,
            self.observation_dim,
            self.action_dim,
            self.reward_dim,
            self.latent_dim,
        ).to(self.device)

        self.latent_encoder_target = LatentEncoder(
            self.use_avg,
            self.observation_dim,
            self.action_dim,
            self.reward_dim,
            self.latent_dim,
        ).to(self.device)

        self.critic = TwinnedQNetwork(
            critic_input_dim if not self.use_critic_preference else critic_input_dim,
            self.action_dim,
            self.reward_dim,
            hidden_units=self.hidden_units,
            use_critic_preference=self.use_critic_preference,
        ).to(self.device)

        self.critic_target = TwinnedQNetwork(
            critic_input_dim if not self.use_critic_preference else critic_input_dim,
            self.action_dim,
            self.reward_dim,
            hidden_units=self.hidden_units,
            use_critic_preference=self.use_critic_preference,
        ).to(self.device)

        self.policy = GaussianPolicy(
            policy_input_dim,
            self.action_dim,
            self.env.action_space,
            hidden_units=self.policy_hidden_units,
        ).to(self.device)

        # Initialize target networks
        self._sync_target_networks()

        # Initialize optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.latent_encoder_optim = optim.Adam(self.latent_encoder.parameters(), lr=self.learning_rate)
        self.q1_optim = optim.Adam(self.critic.Q1.parameters(), lr=self.learning_rate)
        self.q2_optim = optim.Adam(self.critic.Q2.parameters(), lr=self.learning_rate)

        # Entropy tuning
        if self.entropy_tuning:
            self.target_entropy = -self.action_dim
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.alpha = th.tensor(alpha, device=self.device)

        # Replay buffers
        self.replay_buffer = ReplayMemory(
            buffer_size,
            self.observation_shape,
            self.action_shape,
            self.reward_dim,
        )

        self.q_memory = QMemory(self.q_memory_capacity)

        # Training state
        self._n_updates = 0
        self.learning_steps = 0

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def _sync_target_networks(self):
        """Synchronize target networks with current networks."""
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.latent_encoder_target.load_state_dict(self.latent_encoder.state_dict())
        # Disable gradients for target networks
        for param in self.critic_target.parameters():
            param.requires_grad = False
        for param in self.latent_encoder_target.parameters():
            param.requires_grad = False

    def get_config(self) -> dict:
        """Get configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "latent_dim": self.latent_dim,
            "hidden_units": self.hidden_units,
            "policy_hidden_units": self.policy_hidden_units,
            "alpha": self.alpha if not self.entropy_tuning else "auto",
            "entropy_tuning": self.entropy_tuning,
            "learning_starts": self.learning_starts,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.replay_buffer.capacity,
            "q_memory_capacity": self.q_memory_capacity,
            "regular_alpha": self.regular_alpha,
            "regular_bar": self.regular_bar,
            # Config flags
            "use_critic_preference": self.use_critic_preference,
            "use_policy_preference": self.use_policy_preference,
            "policy_use_latent": self.policy_use_latent,
            "policy_use_s": self.policy_use_s,
            "policy_use_w": self.policy_use_w,
            "critic_use_s": self.critic_use_s,
            "critic_use_a": self.critic_use_a,
            "critic_use_both": self.critic_use_both,
            "use_avg": self.use_avg,
            "use_encoder_hardupdate": self.use_encoder_hardupdate,
            "seed": self.seed,
        }

    def save(
        self,
        save_dir: str = "weights/",
        filename: Optional[str] = None,
        save_replay_buffer: bool = False,
    ):
        """Save the agent.

        Args:
            save_dir: Directory to save to
            filename: Filename (without extension)
            save_replay_buffer: Whether to save replay buffer
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            "latent_encoder_state_dict": self.latent_encoder.state_dict(),
            "latent_encoder_target_state_dict": self.latent_encoder_target.state_dict(),
            "latent_encoder_optimizer_state_dict": self.latent_encoder_optim.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "q1_optimizer_state_dict": self.q1_optim.state_dict(),
            "q2_optimizer_state_dict": self.q2_optim.state_dict(),
        }

        if self.entropy_tuning:
            saved_params["log_alpha"] = self.log_alpha
            saved_params["alpha_optimizer_state_dict"] = self.alpha_optim.state_dict()

        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer

        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, os.path.join(save_dir, f"{filename}.tar"))

    def load(self, path: str, load_replay_buffer: bool = True):
        """Load the agent from a file.

        Args:
            path: Path to saved file
            load_replay_buffer: Whether to load replay buffer
        """
        params = th.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(params["policy_state_dict"])
        self.policy_optim.load_state_dict(params["policy_optimizer_state_dict"])
        self.latent_encoder.load_state_dict(params["latent_encoder_state_dict"])
        self.latent_encoder_target.load_state_dict(params["latent_encoder_target_state_dict"])
        self.latent_encoder_optim.load_state_dict(params["latent_encoder_optimizer_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.q1_optim.load_state_dict(params["q1_optimizer_state_dict"])
        self.q2_optim.load_state_dict(params["q2_optimizer_state_dict"])

        if self.entropy_tuning and "log_alpha" in params:
            self.log_alpha = params["log_alpha"]
            self.alpha_optim.load_state_dict(params["alpha_optimizer_state_dict"])

        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch(self):
        """Sample a batch from replay buffer."""
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def _get_preference(self) -> np.ndarray:
        """Sample a random preference vector.

        Returns:
            Random preference vector (sums to 1)
        """
        preference = np.random.rand(self.reward_dim)
        preference = preference.astype(np.float32)
        preference /= preference.sum()
        return preference

    def _build_policy_input(
        self,
        latent: th.Tensor,
        states: Optional[th.Tensor] = None,
        preferences: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Build policy input based on configuration.

        Args:
            latent: Latent representation
            states: Original states (optional)
            preferences: Preferences (optional)

        Returns:
            Policy input tensor
        """
        if self.policy_use_latent:
            inputs = latent
            if self.policy_use_s and states is not None:
                inputs = th.cat([inputs, states], dim=-1)
            if self.policy_use_w and preferences is not None:
                inputs = th.cat([inputs, preferences], dim=-1)
        else:
            # Concat state and preference directly
            inputs = th.cat([states, preferences], dim=-1)
        return inputs

    def _build_critic_input(
        self,
        z_current: th.Tensor,
        z_next: th.Tensor,
        states: Optional[th.Tensor] = None,
        actions: Optional[th.Tensor] = None,
        preferences: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Build critic input based on configuration.

        Args:
            z_current: Current latent representation
            z_next: Predicted next latent representation
            states: Original states (optional)
            actions: Actions (optional)
            preferences: Preferences (optional)

        Returns:
            Critic input tensor
        """
        if self.critic_use_both:
            inputs = th.cat([z_current, z_next], dim=-1)
        else:
            inputs = z_next

        if self.critic_use_s and states is not None:
            inputs = th.cat([inputs, states], dim=-1)
        if self.critic_use_a and actions is not None:
            inputs = th.cat([inputs, actions], dim=-1)

        # Note: preference is handled in TwinnedQNetwork.forward if use_critic_preference

        return inputs

    def calc_target_q(
        self,
        states: th.Tensor,
        preferences: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        next_states: th.Tensor,
        dones: th.Tensor,
    ) -> th.Tensor:
        """Calculate target Q values.

        Args:
            states: Current states
            preferences: Preferences
            actions: Actions
            rewards: Rewards
            next_states: Next states
            dones: Done flags

        Returns:
            Target Q values
        """
        with th.no_grad():
            # Encode next state
            next_latent = self.latent_encoder_target.get_latent_features(
                th.cat([next_states, preferences], dim=-1)
            )

            # Build policy input
            policy_input = self._build_policy_input(next_latent, next_states, preferences)

            # Sample next action
            next_actions, next_entropies, _ = self.policy.sample(policy_input)

            # Get next latent dynamics
            next_sa_z = self.latent_encoder_target.get_dynamic(next_latent, next_actions)

            # Build critic input
            critic_input = self._build_critic_input(
                next_latent, next_sa_z, next_states, next_actions, preferences
            )

            # Get target Q values
            next_q1, next_q2 = self.critic_target(critic_input, next_actions, preferences)

            # Select minimum Q (twin Q trick)
            # Weighted selection based on preferences
            w_q1 = th.einsum("ij,ij->i", next_q1, preferences)
            w_q2 = th.einsum("ij,ij->i", next_q2, preferences)
            mask = (w_q1 < w_q2).unsqueeze(-1).repeat(1, self.reward_dim)
            min_q = th.where(mask, next_q1, next_q2)

            # Add entropy bonus
            next_q = min_q + self.alpha * next_entropies.unsqueeze(-1)

            # Compute target
            target_q = rewards + (1.0 - dones.unsqueeze(-1)) * self.gamma * next_q

        return target_q

    def calc_critic_loss(
        self,
        batch: Tuple[th.Tensor, ...],
        preference: th.Tensor,
        old_critic_state_dict: Optional[dict] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, float, float]:
        """Calculate critic loss with Conflict Objective Regularization.

        Args:
            batch: Batch of (states, preferences, actions, rewards, next_states, dones)
            preference: Current preference vector
            old_critic_state_dict: State dict of old critic for regularization

        Returns:
            Tuple of (q1_loss, q2_loss, td_errors, mean_q1, mean_q2)
        """
        states, batch_prefs, actions, rewards, next_states, dones = batch

        # Repeat preference for batch
        pref_batch = preference.repeat(states.shape[0], 1)

        # Encode current state
        current_latent = self.latent_encoder.get_latent_features(th.cat([states, pref_batch], dim=-1))

        # Get predicted next latent
        pre_next_latent = self.latent_encoder.get_dynamic(current_latent, actions)

        # Build critic input
        critic_input = self._build_critic_input(
            current_latent, pre_next_latent, states, actions, pref_batch
        )

        # Current Q values
        curr_q1, curr_q2 = self.critic(critic_input, actions, pref_batch)

        # Target Q values
        target_q = self.calc_target_q(states, pref_batch, actions, rewards, next_states, dones)

        # TD errors
        td_errors = th.abs(curr_q1.detach() - target_q)
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Base Q loss (weighted by preference)
        q1_loss = th.mean(
            th.einsum("ij,ij->i", (curr_q1 - target_q).pow(2), pref_batch)
        )
        q2_loss = th.mean(
            th.einsum("ij,ij->i", (curr_q2 - target_q).pow(2), pref_batch)
        )

        # Conflict Objective Regularization (COR)
        if old_critic_state_dict is not None and len(self.q_memory) > 0:
            # Create temporary critic with old weights
            old_critic = TwinnedQNetwork(
                self.latent_dim * 2 if self.critic_use_both else self.latent_dim,
                self.action_dim,
                self.reward_dim,
                hidden_units=self.hidden_units,
                use_critic_preference=self.use_critic_preference,
            ).to(self.device)
            old_critic.load_state_dict(old_critic_state_dict)

            # Sample random preference for regularization
            rand_pref = self._get_preference()
            rand_pref_tensor = th.tensor(rand_pref, device=self.device).repeat(states.shape[0], 1)

            # Get Q values from old critic
            old_critic_input = self._build_critic_input(
                current_latent, pre_next_latent, states, actions, rand_pref_tensor
            )
            old_q1, old_q2 = old_critic(old_critic_input, actions, rand_pref_tensor)

            # Regularization loss
            regular_q1 = th.mean((curr_q1 - old_q1.detach()).pow(2))
            regular_q2 = th.mean((curr_q2 - old_q2.detach()).pow(2))

            # Compute stiffness (gradient similarity)
            # Simplified: use Q value difference as proxy
            stiffness_q1 = th.mean(
                F.cosine_similarity(
                    (curr_q1 - target_q).flatten(),
                    (curr_q1 - old_q1.detach()).flatten(),
                    dim=0,
                )
            )
            stiffness_q2 = th.mean(
                F.cosine_similarity(
                    (curr_q2 - target_q).flatten(),
                    (curr_q2 - old_q2.detach()).flatten(),
                    dim=0,
                )
            )

            # Apply regularization only if stiffness is below threshold
            q1_total_loss = q1_loss + max(self.regular_bar - stiffness_q1.item(), 0.0) * self.regular_alpha * regular_q1
            q2_total_loss = q2_loss + max(self.regular_bar - stiffness_q2.item(), 0.0) * self.regular_alpha * regular_q2
        else:
            q1_total_loss = q1_loss
            q2_total_loss = q2_loss

        return q1_total_loss, q2_total_loss, td_errors, mean_q1, mean_q2

    def calc_policy_loss(
        self,
        batch: Tuple[th.Tensor, ...],
        preference: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Calculate policy loss.

        Args:
            batch: Batch of (states, preferences, actions, rewards, next_states, dones)
            preference: Current preference vector

        Returns:
            Tuple of (policy_loss, entropies)
        """
        states, batch_prefs, actions, rewards, next_states, dones = batch
        pref_batch = preference.repeat(states.shape[0], 1)

        # Encode current state
        current_latent = self.latent_encoder.get_latent_features(th.cat([states, pref_batch], dim=-1))

        # Build policy input
        policy_input = self._build_policy_input(current_latent, states, pref_batch)

        # Sample action
        sampled_action, entropy, _ = self.policy.sample(policy_input)

        # Get latent dynamics for sampled action
        sa_z = self.latent_encoder.get_dynamic(current_latent, sampled_action)

        # Build critic input
        critic_input = self._build_critic_input(
            current_latent, sa_z, states, sampled_action, pref_batch
        )

        # Get Q values
        q1, q2 = self.critic(critic_input, sampled_action, pref_batch)

        # Weight by preference and take minimum
        q1_weighted = th.einsum("ij,ij->i", q1, pref_batch)
        q2_weighted = th.einsum("ij,ij->i", q2, pref_batch)
        q = th.min(q1_weighted, q2_weighted)

        # Policy loss: maximize Q + entropy
        policy_loss = th.mean(-q - self.alpha * entropy)

        return policy_loss, entropy

    def calc_entropy_loss(self, entropy: th.Tensor) -> th.Tensor:
        """Calculate entropy tuning loss.

        Args:
            entropy: Policy entropy

        Returns:
            Entropy loss
        """
        # Increase alpha when entropy is below target
        entropy_loss = -th.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
        )
        return entropy_loss

    def update(self):
        """Update all networks."""
        for _ in range(self.gradient_updates):
            self.learning_steps += 1

            # Sample batch
            batch = self._sample_batch()
            states, prefs, actions, rewards, next_states, dones = batch

            # Sample preference for this update
            preference = self._get_preference()
            pref_tensor = th.tensor(preference, device=self.device)

            # Update latent encoder (dynamics prediction)
            with th.no_grad():
                target_latent = self.latent_encoder_target.get_latent_features(
                    th.cat([next_states, prefs], dim=-1)
                )
            current_latent = self.latent_encoder.get_latent_features(th.cat([states, prefs], dim=-1))
            predicted_next_latent = self.latent_encoder.get_dynamic(current_latent, actions)

            dynamic_loss = th.mean((predicted_next_latent - target_latent).pow(2))

            self.latent_encoder_optim.zero_grad()
            dynamic_loss.backward()
            self.latent_encoder_optim.step()

            # Get old critic for regularization
            old_critic_state_dict = None
            if self.learning_steps % self.old_q_update_freq == 0 and len(self.q_memory) > 0:
                old_critic_state_dict = random.choice(self.q_memory.sample())

            # Update critics
            q1_loss, q2_loss, td_errors, mean_q1, mean_q2 = self.calc_critic_loss(
                batch, pref_tensor, old_critic_state_dict
            )

            total_q_loss = self.value_coef * (q1_loss + q2_loss)
            self.q1_optim.zero_grad()
            self.q2_optim.zero_grad()
            total_q_loss.backward()
            self.q1_optim.step()
            self.q2_optim.step()

            # Update policy
            policy_loss, entropy = self.calc_policy_loss(batch, pref_tensor)
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Update alpha if entropy tuning
            if self.entropy_tuning:
                entropy_loss = self.calc_entropy_loss(entropy)
                self.alpha_optim.zero_grad()
                entropy_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()

            # Update target networks
            if self.learning_steps % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            if self.use_encoder_hardupdate:
                if self.learning_steps % self.encoder_update_freq == 0:
                    self.latent_encoder_target.load_state_dict(self.latent_encoder.state_dict())
            else:
                if self.learning_steps % self.encoder_update_freq == 0:
                    polyak_update(
                        self.latent_encoder.parameters(),
                        self.latent_encoder_target.parameters(),
                        self.tau,
                    )

            # Store current critic in Q-memory
            if self.learning_steps % 1000 == 0:
                self.q_memory.append(self.critic.state_dict())

            # Logging
            if self.log and self.learning_steps % 100 == 0:
                tensorboard_log(
                    {
                        "losses/q1_loss": q1_loss.item(),
                        "losses/q2_loss": q2_loss.item(),
                        "losses/policy_loss": policy_loss.item(),
                        "losses/dynamic_loss": dynamic_loss.item(),
                        "metrics/alpha": self.alpha.item(),
                        "metrics/entropy": entropy.mean().item(),
                        "metrics/mean_q1": mean_q1,
                        "metrics/mean_q2": mean_q2,
                        "global_step": self.global_step,
                    }
                )

    @th.no_grad()
    def eval(
        self,
        obs: Union[np.ndarray, th.Tensor],
        w: Union[np.ndarray, th.Tensor],
        torch_action: bool = False,
    ) -> Union[np.ndarray, th.Tensor]:
        """Evaluate policy for given observation and weight.

        Args:
            obs: Observation
            w: Weight/preference vector
            torch_action: Whether to return torch tensor

        Returns:
            Action
        """
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs, dtype=th.float32).unsqueeze(0).to(self.device)
            w = th.tensor(w, dtype=th.float32).unsqueeze(0).to(self.device)

        # Encode state
        if self.policy_use_latent:
            latent = self.latent_encoder.get_latent_features(th.cat([obs, w], dim=-1))
            policy_input = self._build_policy_input(latent, obs, w)
        else:
            policy_input = th.cat([obs, w], dim=-1)

        action = self.policy.get_action(policy_input)

        if not torch_action:
            action = action.squeeze(0).detach().cpu().numpy()

        return action

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        eval_freq: int = 10000,
        reset_num_timesteps: bool = False,
        checkpoints: bool = False,
        save_freq: int = 10000,
    ):
        """Train COLA.

        Args:
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment
            ref_point: Reference point for hypervolume
            known_pareto_front: Known Pareto front (optional)
            num_eval_weights_for_front: Number of weights for front evaluation
            num_eval_episodes_for_front: Episodes per weight evaluation
            num_eval_weights_for_eval: Weights for EUM computation
            eval_freq: Evaluation frequency
            reset_num_timesteps: Reset timestep counter
            checkpoints: Save checkpoints
            save_freq: Checkpoint frequency
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "eval_freq": eval_freq,
                    "reset_num_timesteps": reset_num_timesteps,
                }
            )

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        obs, _ = self.env.reset()

        for step in range(1, total_timesteps + 1):
            self.global_step += 1

            # Sample preference
            preference = self._get_preference()

            # Select action
            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.eval(obs, preference)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.replay_buffer.push(obs, preference, action, reward, next_obs, done)

            obs = next_obs

            # Update networks
            if self.global_step >= self.learning_starts:
                self.update()

            # Reset environment
            if done:
                obs, _ = self.env.reset()
                self.num_episodes += 1

                if self.log and "episode" in info:
                    log_episode_info(info["episode"], np.dot, preference, self.global_step)

            # Evaluation
            if self.log and self.global_step % eval_freq == 0:
                returns_test_tasks = [
                    policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[3]
                    for ew in eval_weights
                ]
                log_all_multi_policy_metrics(
                    current_front=returns_test_tasks,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )

            # Checkpoint
            if checkpoints and self.global_step % save_freq == 0:
                self.save(filename=f"COLA_step_{self.global_step}", save_replay_buffer=False)

        self.close_wandb()
