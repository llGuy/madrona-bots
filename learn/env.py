from madrona_bots import SimManager

def test():
    sim_mgr = SimManager(0, 2048, 69, 32)

    # Train loop
    while True:
        sim_mgr.step()

        action_tensor = sim_mgr.action_tensor().to_torch()
        position_tensor = sim_mgr.position_tensor().to_torch()

        # (num_worlds, num_species)
        species_count = sim_mgr.species_count_tensor().to_torch()
        species_count_cpy = species_count.copy()
        
        # (num_species, 1)
        species_count_cpy.sum(axis=0)
        # Use exclusive prefix sum to get offsets
        offsets = ...

        action_tensor[offsets[2]:offsets[3]]

    sim_mgr.step()

    depth_tensor = sim_mgr.depth_tensor().to_torch()

