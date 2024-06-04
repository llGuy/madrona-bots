import random
import torch
import torch.nn as nn
from pdb import set_trace as T

class A2CNets(nn.Module):
    def __init__(self, feature, recurrent, actor, critic):
        super().__init__()
        self.feature = feature
        self.recurrent = recurrent
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

        recurrent = self._create_random_rlayer()

        # R2D2 (Kapturowski et al. 2019) ZiyuanMa implementation:
        actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)
        )

        a2c_nets = A2CNets(
            nn.Sequential(*all_layers),
            recurrent,
            actor,
            critic
        )

        return a2c_nets

    # Linear layers, TODO: Add more
    def _create_random_llayer(self):
        possible_layers = [
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
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

    # Recurrent layers
    def _create_random_rlayer(self):
        # rnn_input_dim = self.hidden_dim + self.output_dim + 1 # hidden + prev_action + prev_reward; this would require extra caching
        rnn_input_dim = self.hidden_dim
        possible_layers = [
            nn.LSTM(rnn_input_dim, self.hidden_dim),
            nn.GRU(rnn_input_dim, self.hidden_dim),
            nn.RNN(rnn_input_dim, self.hidden_dim)
        ]

        chosen = random.randint(0, len(possible_layers) - 1)

        return possible_layers[chosen]

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, species_generator, device, config=None):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        if config:
            print("Loading model from config")
            self.a2c_nets = self._build_from_config(config).to(device)
        else:
            print("Creating new model")
            self.a2c_nets = species_generator._create_random_net().to(device)
        self.add_module('a2c_nets', self.a2c_nets)

    # TODO: make config system more modular
    def _build_from_config(self, config):
        all_layers = []
        for layer_info in config['layers']:
            if layer_info['type'] == 'linear':
                all_layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
            elif layer_info['type'] == 'activation':
                all_layers.append(getattr(nn, layer_info['activation'])())

        actor_layers = []
        for layer_info in config['actor']:
            if layer_info['type'] == 'linear':
                actor_layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
            elif layer_info['type'] == 'activation':
                actor_layers.append(nn.ReLU())

        critic_layers = []
        for layer_info in config['critic']:
            if layer_info['type'] == 'linear':
                critic_layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
            elif layer_info['type'] == 'activation':
                critic_layers.append(nn.ReLU())

        recurrent_class = getattr(nn, config['recurrent']['type'])
        recurrent = recurrent_class(config['recurrent']['input_dim'], config['recurrent']['hidden_dim'])

        return A2CNets(
            nn.Sequential(*all_layers),
            recurrent,
            nn.Sequential(*actor_layers),
            nn.Sequential(*critic_layers)
        )

    def get_config(self):
        config = {
            'layers': [],
            'actor': [],
            'critic': [],
            'recurrent': {'type': self.a2c_nets.recurrent.__class__.__name__, 'input_dim': self.a2c_nets.recurrent.input_size, 'hidden_dim': self.a2c_nets.recurrent.hidden_size}
        }
        for layer in self.a2c_nets.feature:
            if isinstance(layer, nn.Linear):
                config['layers'].append({'type': 'linear', 'in_features': layer.in_features, 'out_features': layer.out_features})
            elif isinstance(layer, (nn.Tanh, nn.ELU, nn.LogSigmoid, nn.LeakyReLU, nn.ReLU)):
                config['layers'].append({'type': 'activation', 'activation': layer.__class__.__name__})
        
        for layer in self.a2c_nets.actor:
            if isinstance(layer, nn.Linear):
                config['actor'].append({'type': 'linear', 'in_features': layer.in_features, 'out_features': layer.out_features})
            elif isinstance(layer, nn.ReLU):
                config['actor'].append({'type': 'activation', 'activation': 'ReLU'})

        for layer in self.a2c_nets.critic:
            if isinstance(layer, nn.Linear):
                config['critic'].append({'type': 'linear', 'in_features': layer.in_features, 'out_features': layer.out_features})
            elif isinstance(layer, nn.ReLU):
                config['critic'].append({'type': 'activation', 'activation': 'ReLU'})

        return config
    
    def forward(self, state):
        feature_out = self.a2c_nets.feature(state)
        shared_out, _ = self.a2c_nets.recurrent(feature_out)
        action_probs = self.a2c_nets.actor(shared_out)
        value = self.a2c_nets.critic(shared_out)
        return action_probs, value

    def forward_td_zero(self, observations):
        action_probs, values = self.forward(observations.to(self.device))
        actions = torch.distributions.Categorical(logits=action_probs).sample()
        selected_log_probs = action_probs[torch.arange(actions.shape[0]), actions] 

        return actions, selected_log_probs, values
    
    @staticmethod
    def compute_loss(action_log_probs, reward, prev_V, new_V, critic_loss=nn.SmoothL1Loss(), gamma=1.0):
        assert len(action_log_probs) == len(reward) == len(prev_V) == len(new_V)
        advantage = reward + (gamma * new_V.detach().squeeze()) - prev_V.detach().squeeze()
        return -(torch.sum(action_log_probs * advantage)), critic_loss(reward, prev_V.squeeze())