import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import logging as python_logging
LOGGER = python_logging.getLogger()
LOGGER.setLevel(python_logging.INFO)

from absl import logging
logging.set_verbosity(logging.INFO)

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import torch
import tyro


# ──────────────────────────────────────────────────────────────────────────
#  ‣ Torch MLP definition
# ──────────────────────────────────────────────────────────────────────────
class MLP(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: list[int],  # first entry = input dim
        activation: str = "swish",
        activate_final: bool = False,
        bias: bool = True,
        split: bool = False,
    ):
        super().__init__()
        self.act = torch.nn.SiLU() if activation == "swish" else torch.nn.ReLU()
        self.activate_final = activate_final
        self.split = split

        self.hidden = []
        for idx in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[idx], layer_sizes[idx + 1]
            self.hidden.append(torch.nn.Linear(in_dim, out_dim, bias=bias))
        self.hidden = torch.nn.ModuleList(self.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i != len(self.hidden) - 1 or self.activate_final:
                x = self.act(x)
        if self.split:
            loc, _ = torch.chunk(x, 2, dim=-1)
            return torch.tanh(loc)
        return x


# ──────────────────────────────────────────────────────────────────────────
#  ‣ Helpers
# ──────────────────────────────────────────────────────────────────────────
def transfer_weights(jax_params: Mapping, torch_model: torch.nn.Module):
    """
    Copy weights from Brax (JAX) → PyTorch.
    JAX/TF: (in, out)  |  Torch: (out, in)
    """
    idx = 0
    for name, params in jax_params.items():
        if name.startswith("hidden_"):
            layer = torch_model.hidden[int(name.split("_")[-1])]
        elif name.startswith("adapter_"):
            layer = torch_model.adapter[int(name.split("_")[-1])]
        elif name.startswith("conv_"):
            layer = torch_model.conv[int(name.split("_")[-1])]
        else:
            raise ValueError(f"Unexpected parameter name: {name}")
        print(f"Transfer {name}...", layer.weight.shape, layer.bias.shape, params["kernel"].shape, params["bias"].shape)
        layer.weight.data[:] = torch.tensor(np.array(params["kernel"]).T, dtype=torch.float32)
        layer.bias.data[:] = torch.tensor(np.array(params["bias"]), dtype=torch.float32)
        idx += 1
    logging.info("Weights transferred (JAX → Torch) successfully.")


def get_latest_ckpt(path: Path) -> Path | None:
    ckpts = [ckpt for ckpt in path.glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def convert_jax2torch(
    ckpt_dir: Path,
    output_path: str,
    inference_fn,
    policy_params,
    obs_size: int | Mapping[str, tuple[int, ...] | int],
    action_size: int,
    jax_params,
    activation="swish",
):
    # 1 » generate a deterministic random observation batch
    rand_obs = {key: np.random.randn(1, obs_size[key][0]).astype(np.float32) for key in obs_size.keys()}

    # 2 » JAX prediction
    jax_pred, _ = inference_fn(rand_obs, jax.random.PRNGKey(0))
    jax_pred = np.array(jax_pred[0])

    # 3 » Build PyTorch model + load weights

    # only build policy network [1]
    obs_dim = obs_size[policy_params.policy_obs_key][0]
    layer_sizes = [obs_dim] + list(policy_params.policy_hidden_layer_sizes) + [action_size * 2]
    torch_model = MLP(layer_sizes, activation=activation, split=True)
    transfer_weights(jax_params[1]["params"], torch_model)
    torch_model.eval()

    # 4 » Torch prediction
    with torch.no_grad():
        torch_pred = torch_model(torch.from_numpy(rand_obs[policy_params.policy_obs_key])).numpy()[0]

    scripted_model = torch.jit.script(torch_model)
    scripted_model.save(output_path)

    logging.info("Predictions:")
    np.set_printoptions(precision=2, suppress=True)
    logging.info(f"\n\tJAX : {jax_pred}\n\tTorch: {torch_pred}")

    mae = np.mean(np.abs(jax_pred - torch_pred))
    logging.info(f"Mean absolute error (JAX vs Torch): {mae:.2e}")

    np.testing.assert_allclose(jax_pred, torch_pred, rtol=1e-03, atol=1e-05)
    logging.info(f"Success! Torch model saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────────
#  ‣ CLI
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Args:
    exp_name: str


def main(args: Args):
    import json
    from brax.training.agents.ppo.networks import make_ppo_networks
    from src.learning.ppo import train_ppo as ppo
    from src.envs.g1.wrapper import wrap_fn
    from src.envs.g1.g1_tracking_env import G1TrackingEnv, default_config

    ckpt_path = Path(__file__).parent / "experiments" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)
    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint found.")
    logging.info(f"Using checkpoint: {latest_ckpt}")

    output_path = latest_ckpt / "policy.pt"

    task_cfg = default_config()
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    config_path = Path(__file__).parent / "experiments" / args.exp_name / "checkpoints" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    env_cfg.update(config["env_config"])
    policy_cfg.update(config["policy_config"])
    env_cfg.enable_randomize = False
    env_cfg.reference_traj_config.name = {"lafan1": ["dance1_subject1"]}

    env = G1TrackingEnv(terrain_type=env_cfg.terrain_type, config=env_cfg)
    env.prepare_trajectory(env._config.reference_traj_config.name)

    network_factory = functools.partial(make_ppo_networks, **policy_cfg.network_factory)
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=0,
        episode_length=policy_cfg.episode_length,
        normalize_observations=False,
        restore_checkpoint_path=latest_ckpt,
        network_factory=network_factory,
        max_devices_per_host=1,
        num_envs=1,
        wrap_env_fn=wrap_fn,
    )
    make_inference_fn, params, _ = train_fn(environment=env)
    inference_fn = make_inference_fn(params, deterministic=True)

    convert_jax2torch(
        ckpt_dir=latest_ckpt,
        output_path=str(output_path),
        inference_fn=inference_fn,
        policy_params=policy_cfg.network_factory,
        obs_size=env.observation_size,
        action_size=env.action_size,
        jax_params=params,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
