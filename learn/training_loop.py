import numpy as np
import os
import torch
import time
import argparse
import wandb
from ckpt import CheckpointManager
from madrona_bots import SimManager, ScriptBotsViewer
from models import ActorCritic, SpeciesNetGenerator
from util import construct_obs, set_seed
from pdb import set_trace as T


# This is for headless training
class TrainLoopManager:
    def __init__(self, gpu_id, num_worlds, rand_seed, init_num_agents_per_world):
        self.sim_mgr = SimManager(gpu_id, num_worlds, rand_seed, init_num_agents_per_world)

    def loop(self, num_epochs, callable, carry, print_freq=10):
        for relative_epoch in range(1, num_epochs + 1):
            if relative_epoch % print_freq == 0 or relative_epoch == 1:
                print("Relative Epoch ", relative_epoch)
            callable(relative_epoch, carry)

    def get_sim_mgr(self):
        return self.sim_mgr


def train_step(relative_epoch, carry):
    sim_mgr, time_values, args, checkpoint_manager, \
            species_nets, species_optims, \
            best_species_actor_loss, best_species_total_loss, \
            best_species_critic_loss, device, start_epochs = carry

    start = time.time()
    sim_mgr.step()
    end = time.time()
    epoch_duration = end - start
    time_values.append(epoch_duration)
    if args.use_wandb:
        wandb.log({"epoch_fps": args.num_worlds / epoch_duration})

    species_counts = sim_mgr.species_count_tensor().to_torch()
    species_end_offsets = species_counts.sum(dim=0).cumsum(dim=0, dtype=int)
    species_start_offsets = torch.cat((torch.zeros(1, dtype=torch.int, device=device), species_end_offsets[:-1]))

    action_tensor = sim_mgr.action_tensor(False).to_torch()
    memory_tensor = sim_mgr.hidden_state_tensor(False).to_torch()
    all_rewards = sim_mgr.reward_tensor(False).to_torch().clone()
    all_healths = sim_mgr.health_tensor(False).to_torch().clone()

    for sp_idx, (sp_start, sp_end) in enumerate(zip(species_start_offsets, species_end_offsets)):
        if args.verbose:
            print("\nSpecies ", sp_idx + 1)
        epoch = start_epochs[sp_idx] + relative_epoch
        model = species_nets[sp_idx]
        observations = construct_obs(sim_mgr, sp_start, sp_end, prev=False)
        prev_memory = memory_tensor[sp_start:sp_end, :]
        action_probs, new_critic_values, species_memory = model.forward(observations, prev_memory)
        new_memory = model.generate_memory(observations, species_memory)

        distrib = torch.distributions.Categorical(logits=action_probs)
        # Epsilon greedy
        # epsilon = args.init_epsilon  # Probability of choosing a random action
        
        # Independent exploration
        # random_decisions = torch.rand(action_probs.size(0), device=device) < epsilon
        # random_actions = torch.randint(0, args.action_dim, (action_probs.size(0),), device=device)
        # sampled_actions = distrib.sample()
        # actions = torch.where(random_decisions, random_actions, sampled_actions)

        # Dependent exploration (all species agents either explore or sample)
        # if torch.rand(1).item() < epsilon:
        #     actions = torch.randint(0, args.action_dim, (action_probs.size(0),), device=device)
        # else:
        #     actions = distrib.sample()
        
        # No epsilon exploration
        actions = distrib.sample()
        
        print("Average action prob: {:.6f}".format(distrib.log_prob(actions).mean().exp().item()), "Action: ", actions.mode().values.item())
        one_hot_actions = torch.zeros(actions.size(0), args.action_dim, device=device)
        one_hot_actions.scatter_(1, actions.unsqueeze(1), 1)

        optimizer = species_optims[sp_idx]
        rewards = all_rewards[sp_start:sp_end, :]
        prev_observations = construct_obs(sim_mgr, sp_start, sp_end, prev=True)

        og_hidden_state = sim_mgr.hidden_state_tensor(True).to_torch()[sp_start:sp_end, :]
        prev_action_probs, prev_critic_values, _ = model.forward(prev_observations.to(model.device), og_hidden_state)
        # if args.verbose:
            # print("Prev action probs: ", prev_action_probs)
        prev_actions = action_tensor[sp_start:sp_end, :].argmax(dim=1) # extract index from one hot action encoding
        prev_action_log_probs = prev_action_probs[torch.arange(prev_actions.shape[0]), prev_actions]
        actor_loss, critic_loss = model.compute_loss(prev_action_log_probs, rewards.flatten(), prev_critic_values.flatten(), new_critic_values.flatten())

        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        if args.verbose:
            print("Actor: ", actor_loss.item(), "; Critic: ", critic_loss.item())
            print("Total Loss: ", total_loss.item())
        total_loss.backward()
        optimizer.step()

        if args.use_wandb:
            wandb.log({
                f"species_{sp_idx+1}_actor_loss": actor_loss.item(),
                f"species_{sp_idx+1}_critic_loss": critic_loss.item(),
                f"species_{sp_idx+1}_total_loss": total_loss.item(),
                f"species_{sp_idx+1}_count": sp_end - sp_start,
                f"species_{sp_idx+1}_count (per world)": (sp_end - sp_start) / args.num_worlds,
                f"species_{sp_idx+1}_reward": rewards.sum().item(), 
                f"species_{sp_idx+1}_avg_health": all_healths[sp_start:sp_end].mean().item(),
                f"species_{sp_idx+1}_learning_rate": optimizer.param_groups[0]['lr'],
                f"species_{sp_idx+1}_avg_action_prob (taken)": distrib.log_prob(actions).mean().exp().item(),
                f"species_{sp_idx+1}_popular_action (taken)": actions.mode().values.item(),
                f"species_{sp_idx+1}_popular_action (greedy)": torch.argmax(action_probs, dim=1).mode().values.item(),
                f"species_{sp_idx+1}_avg_action_entropy": distrib.entropy().mean().item(),
                "epoch": epoch,
            })
        checkpoint_manager.save(model, optimizer, f"species_{sp_idx+1}", epoch, metric_name='latest', verbose=args.verbose)
        
        if actor_loss < best_species_actor_loss[sp_idx]:
            best_species_actor_loss[sp_idx] = actor_loss
            checkpoint_manager.save(model, optimizer, f"species_{sp_idx+1}", epoch, metric_name='actor_loss', verbose=args.verbose)
        
        if critic_loss < best_species_critic_loss[sp_idx]:
            best_species_critic_loss[sp_idx] = critic_loss
            checkpoint_manager.save(model, optimizer, f"species_{sp_idx+1}", epoch, metric_name='critic_loss', verbose=args.verbose)
        
        if total_loss < best_species_total_loss[sp_idx]:
            best_species_total_loss[sp_idx] = total_loss
            checkpoint_manager.save(model, optimizer, f"species_{sp_idx+1}", epoch, metric_name='total_loss', verbose=args.verbose)
        
        sim_mgr.shift_observations()
        action_tensor[sp_start:sp_end, :] = one_hot_actions.int()
        memory_tensor[sp_start:sp_end, :] = new_memory


def construct_run_name(args):
    # reward_type_id = '0' # first attempt of reward function definition
    # reward_type_id = '_health_loss' # first attempt of reward function definition
    # reward_type_id = '2' # penalty radius, reproduced, ate food, health, 
    # reward_type_id = '3' # only positive rewards: reproduced, hit enemy, ate food (10, 15, 7)
    # reward_type_id = '4' # positive rewards: reproduced, hit enemy, ate food (10, 15, 7); negative: hit ally (-5)
    # reward_type_id = '5' # population health
    # reward_type_id = '6' # population health, ate food (10)
    # reward_type_id = '7' # population health, ate food (10), reproduce (10)
    reward_type_id = '8' # population health, ate food (10), reproduce (10), hit enemy (15) 
    run_name = f"universe_{args.universe_id}-r{reward_type_id}"
    return run_name

def train(args):
    run_name = construct_run_name(args)
    if args.use_wandb:
        wandb.init(project="madrona-bots", name=run_name, config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #sim_mgr = SimManager(0, args.num_worlds, 69, 32)
    gpu_id = 0
    rand_seed = args.seed
    init_num_agents_per_world = 32

    set_seed(rand_seed)
    train_loop_mgr = None

    if args.enable_viewer:
        train_loop_mgr = ScriptBotsViewer(gpu_id, args.num_worlds, \
                rand_seed, init_num_agents_per_world,              \
                1375, 768)
    else:
        train_loop_mgr = TrainLoopManager(gpu_id, args.num_worlds, \
                rand_seed, init_num_agents_per_world)

    base_ckpt_dir = f"{args.model_save_dir}/universe_{args.universe_id}"
    species_nets, species_optims = [], []
    best_species_actor_loss, best_species_critic_loss = [float('inf') for _ in range(args.num_species)], [float('inf') for _ in range(args.num_species)]
    best_species_total_loss = [float('inf') for _ in range(args.num_species)]

    if args.create_universe:
        assert not os.path.exists(base_ckpt_dir), f"Universe {args.universe_id} already exists"
    else:
        assert os.path.exists(base_ckpt_dir), f"Universe {args.universe_id} does not exist"

    checkpoint_manager = CheckpointManager(base_ckpt_dir, restore=True)
    species_generator = SpeciesNetGenerator(args.obs_dim, args.action_dim, args.hidden_dim, args.memory_dim)

    def reinit_fn(config=None):
        return ActorCritic(species_generator, device, config)

    start_epochs = []
    for species_id in range(1, args.num_species + 1):
        if args.create_universe:
            print(f"Creating universe: new model for species {species_id}...")
            model = reinit_fn()
            print(f"Species {species_id} model: ", model.get_config())
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            checkpoint_manager.save(model, optimizer, f"species_{species_id}", 0, metric_name="latest", verbose=True)
            start_epochs.append(0)
        else:
            print(f"Loading cached model for species {species_id}...")
            model, optimizer, loaded_epoch = checkpoint_manager.load(ActorCritic, torch.optim.Adam, f"species_{species_id}", init_fn=reinit_fn, metric_name=args.model_load, verbose=True)
            start_epochs.append(loaded_epoch)
        species_nets.append(model)
        species_optims.append(optimizer)

    time_values = []

    sim_mgr = train_loop_mgr.get_sim_mgr()

    carry = (sim_mgr, time_values, args, checkpoint_manager, \
             species_nets, species_optims, \
             best_species_actor_loss, best_species_total_loss, \
             best_species_critic_loss, device, start_epochs)

    train_loop_mgr.loop(args.num_epochs, train_step, carry)

    np_time_values = np.array(time_values)
    avg_time = np_time_values.mean()
    print(f"Average FPS for simulator: {args.num_worlds / avg_time}")

    if args.use_wandb:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Training loop for species simulation.")
    parser.add_argument('--num_worlds', type=int, default=2048, help='Number of worlds in the simulation')
    parser.add_argument('--universe_id', type=str, default='luc', help='ID for the universe')
    parser.add_argument('--num_species', type=int, default=4, help='Number of species')
    parser.add_argument('--obs_dim', type=int, default=69, help='Observation dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the model')
    parser.add_argument('--action_dim', type=int, default=6, help='Action dimension')
    parser.add_argument('--memory_dim', type=int, default=16, help='RNN hidden state dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--init_epsilon', type=float, default=0.5, help='Random action probability when starting (in epsilon greedy)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights & Biases')
    parser.add_argument('--create_universe', action='store_true', help='Create a new universe')
    parser.add_argument('--model_save_dir', type=str, default='checkpoints', help='Directory to save the model')
    parser.add_argument('--model_load', type=str, default='latest', help='Which model to load')
    parser.add_argument('--enable_viewer', action='store_true', help='Enable visualizer while training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
