from ckpt import CheckpointManager
from madrona_bots import SimManager
import madrona_bots
import os
from pdb import set_trace as T
from models import ActorCritic, SpeciesNetGenerator
from util import create_universe, construct_obs
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sim_mgr = SimManager(0, 2048, 69, 32)
sim_mgr = SimManager(0, 4, 69, 32)

universe_id = 'luc'
num_species = 4
obs_dim = 69 # depth + health + position + semantic + surrounding = 32 + 1 + 2 + 32 + 2
hidden_dim = 128
action_dim = 6
lr = 3e-4

base_ckpt_dir = "checkpoints/universe_" + universe_id
species_nets, species_optims = [], []

if os.path.exists(base_ckpt_dir):
    print("Loading cached model...")
    checkpoint_manager = CheckpointManager(base_ckpt_dir, restore=True)
    species_generator = SpeciesNetGenerator(obs_dim, action_dim, hidden_dim)
    def reinit_fn(config=None):
        return ActorCritic(obs_dim, action_dim, hidden_dim, species_generator, device, config)

    for species_id in range(1, num_species + 1):
        model, optimizer = checkpoint_manager.load(ActorCritic, torch.optim.Adam, f"species_{species_id}", epoch=0, init_fn=reinit_fn)
        species_nets.append(model)
        species_optims.append(optimizer)
else:
    print("Creating new universe!")
    species_nets, species_optims, checkpoint_manager = create_universe(universe_id, num_species, obs_dim, action_dim, hidden_dim, device=device, lr=lr) 

# Train loop
# while True:
for epoch in range(1, 11):
    print("Epoch ", epoch)

    sim_mgr.step()
    species_counts = sim_mgr.species_count_tensor().to_torch()
    species_count_cpy = species_counts.clone()
    species_end_offsets = species_count_cpy.sum(dim=0).cumsum(dim=0)
    species_start_offsets = torch.cat((torch.zeros(1, dtype=torch.int, device=device), species_end_offsets[:-1]))
    
    action_tensor = sim_mgr.action_tensor().to_torch()
    all_rewards = sim_mgr.reward_tensor().to_torch()

    # Loop over species
    for sp_idx, (sp_start, sp_end) in enumerate(zip(species_start_offsets, species_end_offsets)):
        print()
        print("Species ", sp_idx + 1)
        # TODO: Add epoch tracking
        model = species_nets[sp_idx]
        optimizer = species_optims[sp_idx] 

        observations = construct_obs(sim_mgr, sp_start, sp_end)
        rewards = rewards = all_rewards[sp_start:sp_end, :].to(device)

        actions, action_log_probs, critic_values = model.forward_td_zero(observations)
        
        actor_loss, critic_loss = model.compute_loss(action_log_probs, rewards.flatten(), critic_values)
        
        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        print("Actor: ", actor_loss.item(), "; Critic: ", critic_loss.item())
        print("Total Loss: ", total_loss.item())
        total_loss.backward()
        optimizer.step()

        one_hot_actions = torch.zeros(actions.size(0), action_dim, device='cuda:0')
        one_hot_actions.scatter_(1, actions.unsqueeze(1), 1)

        action_tensor[sp_start:sp_end, :] = one_hot_actions.int()
        # T()
        # TODO: add model saving