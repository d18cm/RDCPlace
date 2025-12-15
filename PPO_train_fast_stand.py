import argparse
import os
import time
from datetime import datetime
import csv
import gym
import numpy as np
import torch

# noinspection PyUnresolvedReferences
import place_env
from torch.utils.tensorboard import SummaryWriter

from utils.place_db_bookshelf2 import PlaceDB
from utils.comp_res import comp_res
from utils.save_placement_pl import save_placement
from model.PPO_agent_fast_stand import PPO, Transition


def seed_torch(seed=0):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_ppo():
    """参数解析"""
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    """环境构建"""
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    placedb = PlaceDB(args.benchmark)
    if args.pnm > placedb.node_cnt:
        args.pnm = placedb.node_cnt

    env = gym.make('fast_place-v0', placedb=placedb, num_macros_to_place=args.pnm, grid=args.grid, device=args.device)

    seed_torch(args.seed)
    env.seed(args.seed)

    """目录创建"""
    args.SavingMessage = f"{args.benchmark}_{args.pnm}_{args.seed}_{time_str}_{args.SMInfo}"
    save_fig_path = f"results/{args.SavingMessage}/train_fig"
    save_model_path = f"results/{args.SavingMessage}/train_save_model"
    save_tb_log_path = f"results/{args.SavingMessage}"
    save_pl_path = f"results/{args.SavingMessage}/pl"
    log_file = f"results/{args.SavingMessage}/train_save.csv"

    for folder in [save_fig_path, save_model_path, save_tb_log_path, save_pl_path]:
        os.makedirs(folder, exist_ok=True)

    fwrite = open(log_file, "w")
    fwrite.write("i_epoch,env_reward,running_reward,training_step,hpwl,cost,epoch_time\n")

    with open(f"results/{args.SavingMessage}/train_args.csv", mode='w', newline='') as file:
        writer_args = csv.writer(file)
        writer_args.writerow(['Argument', 'Value'])
        for arg in vars(args):
            writer_args.writerow([arg, getattr(args, arg)])

    writer = SummaryWriter(save_tb_log_path)

    """训练主体"""
    agent = PPO(env, args)
    if args.load_model_path:
        agent.load_param(args.load_model_path)
        print(f"[Agent] : load model success from{args.load_model_path}")
    else:
        print("[Agent] : train from None")

    best_reward = -1e9
    running_reward = -1e9
    last_episode = 0

    for i_epoch in range(args.train_epoch_max):
        epoch_start = time.time()
        state = env.reset()
        done = False
        env_reward = 0.0
        i = 0

        while not done:
            state_tmp = state.clone() if hasattr(state, "clone") else state
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            trans = Transition(state_tmp, action, reward, log_prob, next_state)
            if agent.store_transition(trans):  # full to update
                agent.update(writer)
                update_time = time.time()
                print(f"update time{update_time - epoch_start}")
            env_reward += reward
            state = next_state

        if i_epoch == 0:
            running_reward = env_reward
        running_reward = running_reward * 0.9 + env_reward * 0.1

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)

        print(
            f"[Epoch {i_epoch}] env_reward = {env_reward: .2f}\t"
            f"running_reward = {running_reward: .2f}\t"
            f"hpwl = {hpwl / 1e5: .2f}e5\tcost = {cost: .2f}\t"
            f"epoch_time = {epoch_time: .2f}s"
        )

        env.save_fig(f"{save_fig_path}/{args.benchmark}-{i_epoch}-{hpwl: .2f}.png")

        writer.add_scalar('reward/running_reward', running_reward, i_epoch)
        writer.add_scalar('reward/env_reward', env_reward, i_epoch)
        writer.add_scalar('eval/hpwl', hpwl, i_epoch)
        writer.add_scalar('eval/cost', cost, i_epoch)
        fwrite.write(
            f"{i_epoch}, {env_reward: .2f}, {running_reward: .2f}, {agent.training_step}, {hpwl: .2f}, {cost: .2f}, {epoch_time:.2f}\n")
        fwrite.flush()

        if env_reward > best_reward:
            best_reward = env_reward
            if i_epoch >= 50:
                if os.path.exists(f"{save_model_path}/{last_episode}.pth"):
                    os.remove(f"{save_model_path}/{last_episode}.pth")
                    os.remove(f"{save_pl_path}/{args.benchmark}-{last_episode}.pl")
                last_episode = i_epoch
                agent.save_param(f"{save_model_path}/{i_epoch}.pth")
                try:
                    save_placement(f"{save_pl_path}/{args.benchmark}-{i_epoch}.pl", placedb, env.node_pos, env.ratio)
                except Exception as e:
                    print("Error while saving placement:", e)

    fwrite.close()
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO for macro placement')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--A_lr', type=float, default=1e-3)
    parser.add_argument('--C_lr', type=float, default=1e-4)
    parser.add_argument('--buffer_capacity', type=int, default=5, help="times * npm")
    parser.add_argument('--ppo_epoch', type=int, default=5, help="every ppo update times")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--benchmark', type=str, default='adaptec1', help="benchmark name")
    parser.add_argument('--pnm', type=int, default=2000)
    parser.add_argument('--grid', type=int, default=224)
    parser.add_argument('--train_epoch_max', type=int, default=1000)
    parser.add_argument('--load_model_path', type=str, default=None)

    parser.add_argument('--SMInfo', type=str, default='Debug',help="suffix of the training record provides convenient information for self-viewing")
    return parser.parse_args()


if __name__ == '__main__':
    train_ppo()
