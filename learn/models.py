import random
import torch
import torch.nn as nn
from pdb import set_trace as T

class A2CNets(nn.Module):
    def __init__(self, shared, actor, critic):
        super().__init__()
        self.shared = shared
        self.actor = actor
        self.critic = critic

class SpeciesNetGenerator:
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def _create_random_net(self):
        all_layers = [ nn.Linear(self.input_dim, self.hidden_dim) ]

        num_hidden_layers = random.randint(1, 3)

        for _ in range(num_hidden_layers):
            all_layers.append(self._create_random_llayer())
            all_layers.append(self._create_random_nllayer())

        a2c_nets = A2CNets(
            nn.Sequential(*all_layers), 
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Linear(self.hidden_dim, 1)
        )

        return a2c_nets

    # Linear layers, TODO: Add more
    def _create_random_llayer(self):
        possible_layers = [
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ]

        chosen = random.randint(0, len(possible_layers) - 1)

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

        chosen = random.randint(0, len(possible_layers) - 1)

        return possible_layers[chosen]

class ActorCritic(nn.Module):
    # do we need hidden_dim here?
    def __init__(self, obs_dim, action_dim, hidden_dim, species_generator, device, config=None):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        if config:
            self.a2c_nets = self._build_from_config(config).to(device)
        else:
            self.a2c_nets = species_generator._create_random_net().to(device)
        self.add_module('a2c_nets', self.a2c_nets)

    def _build_from_config(self, config):
        # Method to reconstruct the network from configuration
        all_layers = []
        for layer_info in config['layers']:
            if layer_info['type'] == 'linear':
                all_layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
            elif layer_info['type'] == 'activation':
                all_layers.append(getattr(nn, layer_info['activation'])())
        # reconstruct actor and critic based on saved configuration
        shared = nn.Sequential(*all_layers)
        actor = nn.Linear(config['actor_in'], config['actor_out'])
        critic = nn.Linear(config['critic_in'], config['critic_out'])
        return A2CNets(shared, actor, critic)

    def get_config(self):
        # Method to save the configuration of the network
        config = {'layers': [], 'actor_in': self.a2c_nets.actor.in_features,
                  'actor_out': self.a2c_nets.actor.out_features,
                  'critic_in': self.a2c_nets.critic.in_features,
                  'critic_out': self.a2c_nets.critic.out_features}
        for layer in self.a2c_nets.shared:
            if isinstance(layer, nn.Linear):
                config['layers'].append({'type': 'linear', 'in_features': layer.in_features, 'out_features': layer.out_features})
            elif isinstance(layer, (nn.Tanh, nn.ELU, nn.LogSigmoid, nn.LeakyReLU, nn.ReLU)):
                config['layers'].append({'type': 'activation', 'activation': layer.__class__.__name__})
        return config
    
    def forward(self, state):
        shared_out = self.a2c_nets.shared(state)
        action_probs = self.a2c_nets.actor(shared_out)
        value = self.a2c_nets.critic(shared_out)
        return action_probs, value

    def forward_td_zero(self, observations):
        action_probs, values = self.forward(observations.to(self.device))
        actions = torch.distributions.Categorical(logits=action_probs).sample()
        selected_log_probs = action_probs[torch.arange(actions.shape[0]), actions] 

        return actions, selected_log_probs, values
    
    @staticmethod
    def compute_loss(action_p_vals, G, V, critic_loss=nn.SmoothL1Loss()):
        assert len(action_p_vals) == len(G) == len(V)
        advantage = G - V.detach().squeeze()
        # T()
        return -(torch.sum(action_p_vals * advantage)), critic_loss(G, V.squeeze())
        
# num entities per agents: rows are worlds, columns are species; same thing for rewards