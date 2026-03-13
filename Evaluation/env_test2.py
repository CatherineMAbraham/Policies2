import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import wandb
import argparse
from pathlib import Path

def multiple_envs(model_path,
                  threshold_pos=0.001, 
                  threshold_ori=0.08,
                  maxforce=5,
                  softtissue='spring',
                  n_envs=1,
                  num_eps=100,
                  log=True):
        env_kwargs = {
                'reward_type': 'sparse',
                'max_steps': 100,
                'horizon': 'variable',
                'obs_type': 'dict',
                'distance_threshold_pos': threshold_pos,
                'dt': 0.001,
                'dr':0.01,
                'distance_threshold_ori': threshold_ori,
                'softtissue': softtissue,
                'action_type': 'euler',
                'maxforce': maxforce,
                'start_pos' : 'home',
                'render_mode': None,
                'test': True,}

        env = make_vec_env('gym_fracture:softsurg-v0', n_envs=n_envs, env_kwargs=env_kwargs,vec_env_cls=SubprocVecEnv)
        env = VecNormalize.load(f"{model_path}/vec_normalize.pkl", env) # Register the environment
        env.training = False
        env.norm_reward = False
        model_dir = Path(model_path)
        model_candidates = sorted(
                [p for p in model_dir.glob("model*") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
        )
        if not model_candidates:
                raise FileNotFoundError(f"No model files starting with 'model' found in {model_dir}")

        selected_model = model_candidates[0]
        model = TD3.load(str(selected_model), env=env)
        print(f"Loaded model: {selected_model}")
        dones = []
        contacts = []
        num = num_eps
        episodes_collected = 0
        obs = env.reset()
        while episodes_collected < num:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones_array, info_list = env.step(action)
                
                for i in range(env.num_envs):
                        if dones_array[i]:
                                info = info_list[i]
                                
                                # 1. Get the actual final observation (before the auto-reset)
                                # This is critical if you want to calculate metrics manually
                                final_obs = info.get("terminal_observation")
                                
                                # 2. Get the success flag provided by the environment/Monitor
                                is_success = info.get("is_success", False)
                                
                                # 3. Get your custom 'contact' metric
                                # Note: Ensure your env puts 'contact' in the info dict even on the final step!
                                has_contact = info.get("contact", False)
                                
                                dones.append(is_success)
                                contacts.append(has_contact)
                                
                                episodes_collected += 1
                                print(f"[{episodes_collected}/{num}] Env {i} Success: {is_success} Force: {info.get('force')} Contact: {has_contact}")
                                if log==1:
                                    #table = wandb.Table(data = is_success,columns=["Episode", "Success"])
                                    #histogram = wandb.plot.Histogram(table,value='Success', title="Success Distribution")
                                    wandb.log({"Episode": episodes_collected,  "Contact": has_contact})
                                    wandb.run.summary["final_success_rate"] = sum(dones) / len(dones)
                                if episodes_collected >= num:
                                        break
        success_no_contact = sum(1 for d, c in zip(dones, contacts) if d and not c)
        failure_no_contact = sum(1 for d, c in zip(dones, contacts) if not d and not c)
        success_contact = sum(1 for d, c in zip(dones, contacts) if d and c)
        failure_contact = sum(1 for d, c in zip(dones, contacts) if not d and c)

        print("\nSummary Table:")
        print(f"{'Outcome':<20} {'Count':<10}")
        print(f"{'Success, No Contact':<20} {success_no_contact:<10}")
        print(f"{'Failure, No Contact':<20} {failure_no_contact:<10}")
        print(f"{'Success, Contact':<20} {success_contact:<10}")
        print(f"{'Failure, Contact':<20} {failure_contact:<10}")
        print(f"\nOverall Success Rate: {sum(dones)/len(dones):.2%}")
        if log ==1:
                wandb.run.summary["overall_success_rate"] = sum(dones) / len(dones)
                wandb.run.summary["success_no_contact"] = success_no_contact
                wandb.run.summary["failure_no_contact"] = failure_no_contact
                wandb.run.summary["success_contact"] = success_contact
                wandb.run.summary["failure_contact"] = failure_contact
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained model on multiple environments")
    parser.add_argument("--model_path", type=str, help="Path to the trained model zip file")
    parser.add_argument('--maxforce', type=float, default=4, help='Force threshold for the environment.')
    parser.add_argument('--softtissue', type=str, default="spring", help='Soft Tissue Type.')
    parser.add_argument("--threshold_pos", type=float, default=0.001, help="Position threshold for success")
    parser.add_argument("--threshold_ori", type=float, default=0.08, help="Orientation threshold for success")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments to test on")
    parser.add_argument("--num_eps", type=int, default=100, help="Number of episodes to collect data for")
    parser.add_argument("--log", type=int, default=1, help="Whether to log results to Weights & Biases")
    args = parser.parse_args()
    #remove everything before model in the model path
    model_name = args.model_path.split("/")[-1].split(".")[0]
    if args.log==1:
        wandb.init(project="softsurg", name=f"test_{model_name}")
    multiple_envs(
    model_path=args.model_path,
    maxforce=args.maxforce,
    softtissue=args.softtissue,
    threshold_pos=args.threshold_pos,
    threshold_ori=args.threshold_ori,
    n_envs=args.n_envs,
    num_eps=args.num_eps,
    log=args.log
)
