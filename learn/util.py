import torch
import random
import numpy as np
import os
from ckpt import CheckpointManager
from madrona_bots import SimManager
from models import ActorCritic, SpeciesNetGenerator

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

def construct_obs(sim_mgr: SimManager, start, end, prev=False, verbose=False):
    # i: ending index for species
    if verbose:
        print("Shape of depth tensor: ", sim_mgr.depth_tensor(prev).to_torch()[start:end, :].shape)
        print("Shape of health tensor: ", sim_mgr.health_tensor(prev).to_torch()[start:end, :].shape)
        print("Shape of position tensor: ", sim_mgr.position_tensor(prev).to_torch()[start:end, :].shape)
        print("Shape of semantic tensor: ", sim_mgr.semantic_tensor(prev).to_torch()[start:end, :].shape)
        print("Shape of surrounding tensor: ", sim_mgr.surrounding_tensor(prev).to_torch()[start:end, :].shape)
        
    observations = torch.cat((sim_mgr.depth_tensor(prev).to_torch()[start:end, :],
                        sim_mgr.health_tensor(prev).to_torch()[start:end, :],
                        sim_mgr.position_tensor(prev).to_torch()[start:end, :],
                        sim_mgr.semantic_tensor(prev).to_torch()[start:end, :],
                        sim_mgr.surrounding_tensor(prev).to_torch()[start:end, :],
                       ), dim=1)
    return observations

def create_universe(universe_id, num_species, input_dim, output_dim, hidden_dim, device='cpu', lr=3e-4):
    universe_dir = f"checkpoints/universe_{universe_id}/"
    if os.path.exists(universe_dir):
        print(f"Universe {universe_id} already exists!")
        return
    
    checkpoint_manager = CheckpointManager(base_ckpt_dir=universe_dir, restore=True)
    species_generator = SpeciesNetGenerator(input_dim, output_dim, hidden_dim)

    species_nets, species_optims = [], []
    for species_id in range(1, num_species + 1):
        actor_critic = ActorCritic(obs_dim=input_dim, action_dim=output_dim, hidden_dim=hidden_dim, species_generator=species_generator, device=device)
        optimizer = optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        species_nets.append(actor_critic)
        species_optims.append(optimizer)

        species_dir = f'species_{species_id}/'
        checkpoint_manager.save(actor_critic, optimizer, sub_dir=species_dir, epoch=0)
    
    return species_nets, species_optims, checkpoint_manager

def confirm_load(original_model, loaded_model):
    original_params = original_model.state_dict()
    loaded_params = loaded_model.state_dict()

    for param_name, param_tensor in original_params.items():
        if not torch.equal(param_tensor, loaded_params[param_name]):
            print(f"Mismatch in parameter: {param_name}")
            return False
    print("All parameters match successfully!")
    return True