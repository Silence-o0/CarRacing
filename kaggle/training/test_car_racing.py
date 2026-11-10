"""Evaluate the best CarRacing-v3 checkpoint over several episodes.

Runs the greedy policy (no exploration, no video) and reports reward
statistics across episodes.

Usage:
    python test_car_racing.py [checkpoint] [num_episodes]
"""

import sys

import numpy as np
import torch

from dqn_car_racing import make_env, QNetwork, device


def evaluate_checkpoint(checkpoint="checkpoint_CarRacing-v3_best.pth", num_episodes=10):
    env = make_env()
    policy = QNetwork(a_size=env.action_space.n).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()

    rewards = []
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
        print(f"Episode {ep}\tReward {total:.1f}")

    env.close()
    rewards = np.array(rewards)
    print("-" * 40)
    print(f"Episodes : {num_episodes}")
    print(f"Mean     : {rewards.mean():.1f}")
    print(f"Std      : {rewards.std():.1f}")
    print(f"Min      : {rewards.min():.1f}")
    print(f"Max      : {rewards.max():.1f}")
    return rewards


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_CarRacing-v3_best.pth"
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    evaluate_checkpoint(checkpoint=checkpoint, num_episodes=num_episodes)