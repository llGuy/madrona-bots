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
    def __init__(self, num_init_species, input_dim, output_dim, hidden_dim):
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
    def __init__(self, obs_dim, action_dim, hidden_dim, species_generator, device):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.a2c_nets = species_generator._create_random_net().to(device)
        self.add_module('a2c_nets', self.a2c_nets)
        self.device = device
    
    def forward(self, state):
        shared_out = self.a2c_nets.shared(state)
        action_probs = self.a2c_nets.actor(shared_out)
        value = self.a2c_nets.critic(shared_out)
        return action_probs, value

    def train_env_episode(self, env, gamma=.99, render=False):
        rewards = []
        critic_vals = []
        action_lp_vals = []

        observation, _ = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            observation = torch.from_numpy(observation).to(self.device)

            action_probs, value = self.forward(observation)

            action = torch.distributions.Categorical(logits=action_probs).sample()
            action_log_prob = action_probs[action]

            critic_vals.append(value)
            action_lp_vals.append(action_log_prob)

            observation, reward, terminated, truncated, info = env.step(action.item())
            done = truncated or terminated
            rewards.append(torch.tensor(reward).double())

        total_reward = sum(rewards)

        for t_i in range(len(rewards)):
            G = 0
            for t in range(t_i, len(rewards)):
                G += rewards[t] * (gamma ** (t - t_i))
            rewards[t_i] = G

        rewards = torch.stack(rewards)
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + .000000000001)

        return rewards, torch.stack(critic_vals), torch.stack(action_lp_vals), total_reward

    @staticmethod
    def compute_loss(action_p_vals, G, V, critic_loss=nn.SmoothL1Loss()):
        assert len(action_p_vals) == len(G) == len(V)
        advantage = G - V.detach().squeeze()
        return -(torch.sum(action_p_vals * advantage)), critic_loss(G, V.squeeze())
        

# num entities per agents: rows are worlds, columns are species; same thing for rewards