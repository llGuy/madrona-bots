import numpy as np
from ckpt import CheckpointManager
from madrona_bots import SimManager
import madrona_bots
import os
from pdb import set_trace as T
from models import ActorCritic, SpeciesNetGenerator
from util import create_universe, construct_obs
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_worlds = 2048
sim_mgr = SimManager(0, num_worlds, 69, 32)

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

time_values = []

# Train loop
# while True:
for epoch in range(1, 101):
    print("Epoch ", epoch)

    start = time.time()
    sim_mgr.step()
    end = time.time()

    time_values.append(end - start)
    
    species_counts = sim_mgr.species_count_tensor().to_torch()
    species_end_offsets = species_counts.sum(dim=0).cumsum(dim=0, dtype=int)
    species_start_offsets = torch.cat((torch.zeros(1, dtype=torch.int, device=device), species_end_offsets[:-1]))
    
    action_tensor = sim_mgr.action_tensor(False).to_torch()
    all_rewards = sim_mgr.reward_tensor(False).to_torch().clone()

    # Loop over species
    for sp_idx, (sp_start, sp_end) in enumerate(zip(species_start_offsets, species_end_offsets)):
        print()
        print("Species ", sp_idx + 1)
        # TODO: Add epoch tracking
        model = species_nets[sp_idx]
        # Loss on Previous (s, a, r):
        if epoch > 1:
            optimizer = species_optims[sp_idx]

            rewards = all_rewards[sp_start:sp_end, :]
            prev_observations = construct_obs(sim_mgr, sp_start, sp_end, prev=True)

            prev_action_probs, critic_values = model.forward(prev_observations.to(model.device))
            prev_actions = action_tensor[sp_start:sp_end, :].argmax(dim=1) # get actions from previous timestep
            prev_action_log_probs = prev_action_probs[torch.arange(prev_actions.shape[0]), prev_actions]
            # _, prev_action_log_probs, critic_values = model.forward_td_zero(prev_observations)
            actor_loss, critic_loss = model.compute_loss(prev_action_log_probs, rewards.flatten(), critic_values.flatten())
            
            optimizer.zero_grad()
            total_loss = actor_loss + critic_loss
            print("Actor: ", actor_loss.item(), "; Critic: ", critic_loss.item())
            print("Total Loss: ", total_loss.item())
            total_loss.backward()
            optimizer.step()

        # Determine next action
        observations = construct_obs(sim_mgr, sp_start, sp_end, prev=False)
        shared_out = model.a2c_nets.shared(observations)
        action_probs = model.a2c_nets.actor(shared_out)
        actions = torch.distributions.Categorical(logits=action_probs).sample()

        one_hot_actions = torch.zeros(actions.size(0), action_dim, device='cuda:0')
        one_hot_actions.scatter_(1, actions.unsqueeze(1), 1)
        
        sim_mgr.shift_observations()
        action_tensor[sp_start:sp_end, :] = one_hot_actions.int()
        # TODO: add model saving

np_time_values = np.array(time_values[1:])
avg_time = np_time_values.mean()

print(f"Average FPS for simulator: {num_worlds / avg_time}")