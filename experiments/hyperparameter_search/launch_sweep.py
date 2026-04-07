import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import mo_gymnasium as mo_gym
import numpy as np
import yaml
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import seed_everything
from morl_baselines.common.experiments import (
    ALGOS,
    ENVS_WITH_KNOWN_PARETO_FRONT,
    StoreDict,
)
from morl_baselines.common.tensorboard_logger import (
    TensorBoardLogger,
    init as tensorboard_init,
    log as tensorboard_log,
    finish as tensorboard_finish,
)
from morl_baselines.common.utils import reset_tensorboard_env


@dataclass
class WorkerInitData:
    experiment_id: str
    seed: int
    config: dict
    worker_num: int


@dataclass
class WorkerDoneData:
    hypervolume: float
    log_dir: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="Name of the algorithm to run", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env-id", type=str, help="MO-Gymnasium id of the environment to run", required=True)
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True
    )

    parser.add_argument("--tensorboard-logdir", type=str, help="TensorBoard log directory", default="runs")
    parser.add_argument("--project-name", type=str, help="Project name to use for the sweep", default="MORL-Baselines")

    parser.add_argument("--sweep-count", type=int, help="Number of trials to do in the sweep worker", default=10)
    parser.add_argument("--num-seeds", type=int, help="Number of seeds to use for the sweep", default=3)

    parser.add_argument(
        "--seed", type=int, help="Random seed to start from, seeds will be in [seed, seed+num-seeds)", default=10
    )

    parser.add_argument(
        "--train-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the train method algorithm. Example: --train-hyperparams num_eval_weights_for_front:10 timesteps_per_iter:10000",
        default={},
    )

    parser.add_argument(
        "--config-name",
        type=str,
        help="Name of the config to use for the sweep, defaults to using the same name as the algorithm.",
    )

    args = parser.parse_args()

    if not args.config_name:
        args.config_name = f"{args.algo}.yaml"
    elif not args.config_name.endswith(".yaml"):
        args.config_name += ".yaml"

    return args


def train(worker_data: WorkerInitData) -> WorkerDoneData:
    # Reset the TensorBoard environment variables
    reset_tensorboard_env()

    seed = worker_data.seed
    experiment_id = worker_data.experiment_id
    config = worker_data.config
    worker_num = worker_data.worker_num

    # Set the seed
    seed_everything(seed)

    if args.algo == "pgmorl":
        # PGMORL creates its own environments because it requires wrappers
        print(f"Worker {worker_num}: Seed {seed}. Instantiating {args.algo} on {args.env_id}")
        eval_env = mo_gym.make(args.env_id)
        algo = ALGOS[args.algo](
            env_id=args.env_id,
            origin=np.array(args.ref_point),
            tensorboard_log=True,
            **config,
            seed=seed,
            experiment_id=experiment_id,
        )

        # Launch the agent training
        print(f"Worker {worker_num}: Seed {seed}. Training agent...")
        algo.train(
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=None,
            **args.train_hyperparams,
        )

    else:
        print(f"Worker {worker_num}: Seed {seed}. Instantiating {args.algo} on {args.env_id}")
        env = MORecordEpisodeStatistics(mo_gym.make(args.env_id), gamma=config["gamma"])
        eval_env = mo_gym.make(args.env_id)

        algo = ALGOS[args.algo](env=env, tensorboard_log=True, **config, seed=seed, experiment_id=experiment_id)

        if args.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
            known_pareto_front = env.unwrapped.pareto_front(gamma=config["gamma"])
        else:
            known_pareto_front = None

        # Launch the agent training
        print(f"Worker {worker_num}: Seed {seed}. Training agent...")
        algo.train(
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=known_pareto_front,
            **args.train_hyperparams,
        )

    # Get the hypervolume from the TensorBoard run (stored in the logger)
    # For TensorBoard, we store final metrics in a summary file
    hypervolume = config.get("_final_hypervolume", 0.0)
    print(f"Worker {worker_num}: Seed {seed}. Hypervolume: {hypervolume}")

    return WorkerDoneData(hypervolume=hypervolume, log_dir=config.get("_log_dir", ""))


def main():
    # For TensorBoard, we don't have a central sweep runner
    # Each worker runs independently and logs to its own directory

    # Spin up workers to run experiments in parallel
    with ProcessPoolExecutor(max_workers=args.num_seeds) as executor:
        futures = []
        for num in range(args.num_seeds):
            seed = seeds[num]
            config = dict(sweep_config.get("parameters", {}))
            # Extract values from sweep config format if present
            for key, value in config.items():
                if isinstance(value, dict) and "value" in value:
                    config[key] = value["value"]
            config["_log_dir"] = f"runs/{args.project_name}/{args.algo}/seed_{seed}"
            futures.append(
                executor.submit(
                    train, WorkerInitData(experiment_id=args.algo, seed=seed, config=config, worker_num=num)
                )
            )

        # Get results from workers
        results = [future.result() for future in futures]

    # Get the hypervolume from the results
    hypervolume_metrics = [result.hypervolume for result in results]
    print(f"Hypervolumes of the experiment: {hypervolume_metrics}")

    # Compute the average hypervolume
    average_hypervolume = sum(hypervolume_metrics) / len(hypervolume_metrics)
    print(f"Average hypervolume: {average_hypervolume}")

    # Print log directories for TensorBoard
    print(f"Log directories: {[result.log_dir for result in results]}")
    print(f"Run 'tensorboard --logdir runs/{args.project_name}/{args.algo}' to view results")


args = parse_args()

# Create an array of seeds to use for the sweep
seeds = [args.seed + i for i in range(args.num_seeds)]

# Load the sweep config
config_file = os.path.join(os.path.dirname(__file__), "configs", args.config_name)

# Set up the default hyperparameters
with open(config_file) as file:
    sweep_config = yaml.safe_load(file)

print(f"Running hyperparameter search for {args.algo} on {args.env_id}")
print(f"Configuration loaded from: {config_file}")
print(f"Results will be logged to: {args.tensorboard_logdir}")
print(f"Number of seeds: {args.num_seeds}")
print(f"Seeds: {seeds}")

# For TensorBoard, we run the experiments directly without sweep automation
# Each seed will be logged to a separate directory
for _ in range(args.sweep_count):
    main()
