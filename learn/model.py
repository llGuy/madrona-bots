import random
import torch.nn as nn

@dataclass
class A2CNets:
    shared: nn.Sequential
    actor: nn.Linear
    critic: nn.Linear

class SpeciesNetGenerator:
    def __init__(self, num_init_species, input_dim, output_dim, inner_layer_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_layer_dim = inner_layer_dim

    def _create_random_net(self):
        all_layers = [ nn.Linear(self.input_dim, self.inner_layer_dim) ]

        num_hidden_layers = random.randint(1, 3)

        for _ in range(num_hidden_layers):
            all_layers.append(_create_random_llayer())
            all_layers.append(_create_random_nllayer())

        a2c_nets = A2CNets(
            *all_layers, 
            nn.Linear(self.inner_layer_dim, self.output_dim),
            nn.Linear(self.inner_layer_dim, 1)
        )

        return a2c_nets

    # Linear layers, TODO: Add more
    def _create_random_llayer(self):
        possible_layers = [
            nn.Linear(self.inner_layer_dim, self.inner_layer_dim)
        ]

        chosen = random.randint(len(possible_layers))

        return possible_layers[chosen]

    # Non-linear layers
    def _create_random_nllayer(self):
        possible_layers = [
            nn.Tanh(),
            nn.ELU(),
            nn.LogSigmoid(),
            nn.LeakyReLU(),
            nn.ReLU()
        ]

        chosen = random.randint(len(possible_layers))

        return possible_layers[chosen]

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = 
