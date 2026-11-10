"""Deep Q-Learning on CarRacing-v3 (Box2D).

Environment: https://gymnasium.farama.org/environments/box2d/car_racing/

Observations are 96x96x3 RGB frames. We convert to grayscale, resize to
84x84 and stack 4 consecutive frames so the network can perceive motion.
The discrete action space has 5 actions: do-nothing, steer-left,
steer-right, gas, brake.
"""

from collections import deque
from pathlib import Path
import os
import subprocess
import sys


if __name__ == "__main__" and Path("/kaggle").is_dir():
    environment = Path("/kaggle/working/car-racing-py312")
    environment_python = environment / "bin/python"
    if Path(sys.prefix).resolve() != environment.resolve():
        if not environment_python.is_file():
            print("Creating an isolated Python 3.12 environment...", flush=True)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "uv",
            ])
            subprocess.check_call([
                sys.executable, "-m", "uv", "venv",
                "--python", "3.12", "--seed", str(environment),
            ])
        child_environment = os.environ.copy()

        child_environment["PATH"] = (
            str(environment_python.parent) + os.pathsep
            + child_environment.get("PATH", "")
        )
        result = subprocess.run([
            str(environment_python), str(Path(__file__).resolve()), *sys.argv[1:],
        ], env=child_environment)
        raise SystemExit(result.returncode)

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The CarRacing environment must use Python 3.12.")
    print("Python:", sys.version, flush=True)
    print("Installing CarRacing dependencies with local versions...", flush=True)

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "swig==4.2.1", "setuptools==84.2.1", "wheel==0.45.1",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.8.0+cu128",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-build-isolation",
        "numpy==1.26.4",
        "gymnasium==1.3.0",
        "pygame-ce==2.5.8",
        "moviepy==2.2.1",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
        "pillow==11.3.0",
        "box2d-py==2.3.5",
        "opencv-python==4.10.0.84",
    ])

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    RecordVideo,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # autotune convs for fixed input size

FRAME_STACK = 4
FRAME_SIZE = 84


VALIDATION_SEEDS = tuple(range(10000, 10010))


def make_env(render_mode=None):
    """CarRacing with grayscale, resize and frame-stacking wrappers."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    env = FrameStackObservation(env, FRAME_STACK)
    return env


def make_vec_env(num_envs):
    """Several CarRacing instances stepped in parallel subprocesses."""
    return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])


class QNetwork(nn.Module):
    """CNN mapping a stack of frames to action values (Nature DQN)."""

    def __init__(self, in_channels=FRAME_STACK, a_size=5):
        super().__init__()
        self.a_size = a_size

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, a_size)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.a_size)
        state = torch.tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values, dim=1).item()

    def act_batch(self, states, epsilon=0.0):
        """Epsilon-greedy actions for a batch of states (one per env)."""
        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = self.forward(states).argmax(dim=1).cpu().numpy()
        explore = np.random.rand(len(actions)) < epsilon
        actions[explore] = np.random.randint(self.a_size, size=int(explore.sum()))
        return actions


class ReplayBuffer:
    """Preallocated ring buffer. Frames stay uint8 to cut memory and
    host->GPU transfer 4x; the network casts to float on device."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.from_numpy(self.states[idx]).to(device, non_blocking=True)
        next_states = torch.from_numpy(self.next_states[idx]).to(device, non_blocking=True)
        actions = torch.from_numpy(self.actions[idx]).to(device).unsqueeze(1)
        rewards = torch.from_numpy(self.rewards[idx]).to(device).unsqueeze(1)
        dones = torch.from_numpy(self.dones[idx]).to(device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, action_dim, obs_shape, lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, buffer_size=50000,
                 batch_size=128, target_update=1000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.model = QNetwork(a_size=action_dim).to(device)
        self.target_model = QNetwork(a_size=action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, obs_shape)
        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        self.learn_steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        with torch.autocast(device.type, enabled=device.type == "cuda"):
            q_values = self.model(states).gather(1, actions)
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1, keepdim=True)[0]
                target = rewards + self.gamma * next_q * (1 - dones)
            loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(num_envs=8, n_episodes=1000, updates_per_step=2, validate_every=50):
    """Train with several CarRacing envs stepped in parallel.

    Transitions from every env feed a single shared replay buffer and one
    Q-network. Gymnasium's vector API auto-resets a sub-env on the step
    after it finishes, so we skip storing that bridging transition.
    """
    env = make_vec_env(num_envs)
    agent = DQNAgent(env.single_action_space.n, env.single_observation_space.shape)

    scores = []
    scores_deque = deque(maxlen=100)
    best_reward = -np.inf
    last_validation_episode = 0  # added
    checkpoint = "checkpoint_CarRacing-v3_validation_best.pth"

    states, _ = env.reset()
    prev_done = np.zeros(num_envs, dtype=bool)
    episode_rewards = np.zeros(num_envs)
    completed = 0

    while completed < n_episodes:
        actions = agent.model.act_batch(states, epsilon=agent.epsilon)
        next_states, rewards, terminations, truncations, _ = env.step(actions)
        dones = np.logical_or(terminations, truncations)

        for i in range(num_envs):
            if prev_done[i]:
                continue  # autoreset step: next_states[i] is a fresh episode start
            agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
            episode_rewards[i] += rewards[i]
            if dones[i]:
                score = episode_rewards[i]
                scores.append(score)
                scores_deque.append(score)
                episode_rewards[i] = 0
                completed += 1
                agent.decay_epsilon()
                avg = np.mean(scores_deque)
                print(f"Episode {completed}\tScore {score:.1f}\tAvg {avg:.1f}\tEps {agent.epsilon:.3f}")


                if completed % validate_every == 0:
                    validation_reward = evaluate(agent.model, completed)  # added
                    last_validation_episode = completed  # added
                    if validation_reward > best_reward:  # added
                        best_reward = validation_reward  # added
                        torch.save(agent.model.state_dict(), checkpoint)  # added
                        print(f"  Saved checkpoint (validation mean {best_reward:.1f}) -> {checkpoint}")  # added

        states = next_states
        prev_done = dones

        for _ in range(updates_per_step):
            agent.replay()

    env.close()


    if completed != last_validation_episode:  # added
        validation_reward = evaluate(agent.model, completed)  # added
        if validation_reward > best_reward:  # added
            best_reward = validation_reward  # added
            torch.save(agent.model.state_dict(), checkpoint)  # added
            print(f"  Saved checkpoint (validation mean {best_reward:.1f}) -> {checkpoint}")  # added
    return scores


def test(env, policy, num_episodes=5):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        for _ in range(10000):  # max steps per episode
            action = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward


            if done:  # added
                break  # added
    avg = total_reward / num_episodes
    print(f"Total Reward: {avg:.1f}")
    return avg


def evaluate(policy, episode, video_folder="videos/validation_best"): 
    """Greedy validation episode recorded to MP4 (online rendering)."""
    was_training = policy.training

    random_state = np.random.get_state() # added
    env = None 
    try:
        policy.eval()
        env = make_env(render_mode="rgb_array") 
        env = RecordVideo( 
            env, 
            video_folder=video_folder, 
            name_prefix=f"val-ep{episode}", 
            episode_trigger=lambda e: e == 0, 
        )  
        rewards = []  
        for seed in VALIDATION_SEEDS:  
            state, _ = env.reset(seed=seed) 
            done = False  
            total_reward = 0.0  
            while not done: 

                action = policy.act(state, epsilon=0.0) 
                state, reward, terminated, truncated, _ = env.step(action)  
                done = terminated or truncated 
                total_reward += reward  
            rewards.append(total_reward)  
            print(f"  Validation seed {seed}: reward {total_reward:.1f}") 
    finally: 

        try: 
            if env is not None: 
                env.close()  
        finally:  
            policy.train(was_training) 
            np.random.set_state(random_state) 

    mean_reward = float(np.mean(rewards))  # added
    print(f"  Validation @ episode {episode}: mean {mean_reward:.1f}, " 
          f"std {np.std(rewards):.1f}, episodes {len(rewards)} " 
          f"(video -> {video_folder}/)") 
    return mean_reward 


def render(checkpoint="checkpoint_CarRacing-v3_validation_best.pth", num_episodes=3, 
           video_folder="videos/validation_best"): 
    """Record the trained agent playing to MP4 files (works headless)."""
    env = make_env(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="car_racing",
        episode_trigger=lambda e: True,
    )
    policy = QNetwork(a_size=env.action_space.n).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()
    test(env, policy, num_episodes=num_episodes)
    env.close()
    print(f"Saved videos to {video_folder}/")


if __name__ == "__main__":
    print("device:", device)
    train(num_envs=8)
    render()
