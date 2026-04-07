# COLA: Conflict Objective Regularization in Latent Space

```{eval-rst}
.. autoclass:: morl_baselines.multi_policy.cola.cola.COLA
    :members:
```

## Overview

COLA (Conflict Objective Regularization in Latent Space) is a general-policy multi-objective reinforcement learning algorithm that learns in a shared latent space and mitigates optimization conflicts across different preferences.

The algorithm was introduced in:
> **COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space**  
> Pengyi Li, Hongyao Tang, Yifu Yuan, Jianye Hao, Zibin Dong, and Yan Zheng  
> *Advances in Neural Information Processing Systems (NeurIPS) 2025*

## Key Features

COLA addresses two key challenges in general-policy MORL:

### 1. Objective-agnostic Latent Dynamics Model (OADM)

OADM builds a shared latent space that captures environment dynamics through temporal consistency, enabling efficient knowledge sharing across diverse preferences. The latent encoder:
- Maps state-preference pairs to a compact latent representation
- Learns to predict latent dynamics (next latent state given action)
- Shares latent representations across all objectives

### 2. Conflict Objective Regularization (COR)

COR regularizes value updates when optimization directions under different preferences conflict, stabilizing value approximation and improving policy learning. It:
- Measures gradient conflict between different preference optimizations
- Applies regularization only when conflict exceeds a threshold
- Stores historical Q-networks for regularization references

## Architecture

COLA adopts **Envelope SAC** as its backbone algorithm and conditions both value and policy networks on preferences to cover the entire preference space.

```
State + Preference → Latent Encoder → Latent Representation
                                              ↓
                    ┌─────────────────────────┼─────────────────────────┐
                    ↓                         ↓                         ↓
            Dynamics Predictor        Policy Network            Twin Q-Networks
            (z, a) → z'_pred          (z) → π(a|z,w)           (z, a, w) → Q(s,a,w)
```

## Algorithm Details

### Networks

- **LatentEncoder**: Encodes (state, preference) pairs into latent space and predicts latent dynamics
- **GaussianPolicy**: SAC-style policy that outputs mean and log_std of action distribution
- **TwinnedQNetwork**: Twin Q-networks for stable learning, operating on latent space

### Training Loop

1. Sample random preference vector w
2. Interact with environment using current policy
3. Store transitions in replay buffer
4. Update networks:
   - Latent encoder: minimize dynamics prediction error
   - Critic: minimize TD error with COR regularization
   - Policy: maximize expected Q-value with entropy bonus
   - Alpha: automatic entropy tuning (optional)

### Configuration Options

COLA provides several configuration flags to control network architectures:

| Flag | Description |
|------|-------------|
| `use_critic_preference` | Condition Q-networks on preferences |
| `use_policy_preference` | Condition policy on preferences |
| `policy_use_latent` | Use latent space as policy input |
| `policy_use_s` | Include raw state in policy input (with latent) |
| `policy_use_w` | Include preference in policy input (with latent) |
| `critic_use_s` | Include raw state in Q-network input |
| `critic_use_a` | Include action in Q-network input |
| `critic_use_both` | Use both current and next latent in Q input |
| `use_avg` | Use L1 normalization on latent features |
| `use_encoder_hardupdate` | Use hard updates for encoder target |

## Usage Example

```python
import mo_gymnasium as mo_gym
import numpy as np
from morl_baselines.multi_policy.cola import COLA

# Create environment
env = mo_gym.make("mo-hopper-v5", cost_objective=False, max_episode_steps=500)
eval_env = mo_gym.make("mo-hopper-v5", cost_objective=False, max_episode_steps=500)

# Initialize COLA agent
agent = COLA(
    env,
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    buffer_size=int(1e6),
    batch_size=256,
    latent_dim=50,
    regular_alpha=0.1,
    regular_bar=0.25,
    log=True,
    seed=0,
)

# Train the agent
agent.train(
    total_timesteps=1_000_000,
    eval_env=eval_env,
    ref_point=np.array([0.0, 0.0]),
    eval_freq=10_000,
    num_eval_weights_for_front=100,
)
```

## Hyperparameters

Default hyperparameters based on the original paper:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Learning rate for all optimizers |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update coefficient |
| `batch_size` | 256 | Batch size for training |
| `latent_dim` | 50 | Dimension of latent space |
| `hidden_units` | [256, 256] | Hidden units for Q-networks |
| `policy_hidden_units` | [128, 128] | Hidden units for policy |
| `regular_alpha` | 0.1 | COR regularization strength |
| `regular_bar` | 0.25 | COR stiffness threshold |
| `learning_starts` | 10000 | Steps before learning starts |

## Supported Environments

COLA is designed for continuous control tasks with multiple objectives:

- **2D tasks**: `MO-HalfCheetah-v5`, `MO-Hopper-v5`, `MO-Walker2d-v5`, `MO-Ant-v5` (2 objectives)
- **3D tasks**: `MO-Hopper-v5`, `MO-Ant-v5` (3 objectives)
- **4D tasks**: `MO-Ant-v5` (4 objectives)
- **5D tasks**: `MO-HalfCheetah-v5`, `MO-Hopper-v5`, `MO-Ant-v5` (5 objectives)

## Performance

COLA exhibits higher sample efficiency and better final hypervolume/utility compared to other general-policy methods on multi-objective continuous control tasks (2-5 objectives), owing to:
- OADM enabling efficient knowledge sharing across preferences
- COR providing conflict-aware value learning

## Citation

If you use COLA in your research, please cite:

```bibtex
@inproceedings{Li2025COLA,
  title     = {COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space},
  author    = {Pengyi Li and Hongyao Tang and Yifu Yuan and Jianye Hao and Zibin Dong and Yan Zheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=Cldpn7H3NN}
}
```

## Implementation Notes

This implementation follows the morl-baselines architecture and coding conventions:
- Inherits from `MOAgent` and `MOPolicy` base classes
- Uses shared utilities from `morl_baselines.common`
- Compatible with wandb logging
- Supports checkpointing and resume training
