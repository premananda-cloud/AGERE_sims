"""
weight_manager/export_npz.py

Exports a trained SB3 PPO checkpoint's policy network into a plain
numpy .npz file -- no stable-baselines3 / torch / gymnasium dependency
needed to *run* the resulting file, only to produce it. Meant for
deploying a policy to constrained targets (Raspberry Pi, etc.) where
installing the full training stack isn't practical.

Was model/extract_weight.py (a one-off script hardcoded to
hover_champion.zip); generalized here to take --model/--out so any
checkpoint can be exported, and moved into weight_manager/ since "package
a trained artifact for deployment" is a checkpoint-lifecycle concern like
the rest of this package, not something specific to the hover task.

Only supports SB3's default MlpPolicy shape (Sequential policy_net with
Tanh activations, indices 0 and 2 being the two Linear layers -- i.e.
net_arch=[64,64] or similar single-list net_arch, not a dict split
between pi/vf). If a checkpoint uses a custom net_arch depth or a
pi/vf-split net_arch, this will raise a clear KeyError on the state_dict
lookup rather than silently exporting the wrong thing -- check
ppo_config.net_arch for that checkpoint's training run if it does.
"""

import argparse

import numpy as np
from stable_baselines3 import PPO


def export(model_path: str, out_path: str):
    model = PPO.load(model_path, device="cpu")
    sd = model.policy.state_dict()

    W1 = sd["mlp_extractor.policy_net.0.weight"].numpy()
    b1 = sd["mlp_extractor.policy_net.0.bias"].numpy()
    W2 = sd["mlp_extractor.policy_net.2.weight"].numpy()
    b2 = sd["mlp_extractor.policy_net.2.bias"].numpy()
    Wa = sd["action_net.weight"].numpy()
    ba = sd["action_net.bias"].numpy()

    np.savez(out_path, W1=W1, b1=b1, W2=W2, b2=b2, Wa=Wa, ba=ba)
    print(f"Exported to {out_path}")
    print("shapes:", W1.shape, b1.shape, W2.shape, b2.shape, Wa.shape, ba.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a checkpoint .zip")
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()
    export(args.model, args.out)


if __name__ == "__main__":
    main()
