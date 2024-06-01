# from torch import optim

# def train(dev, sim, cfg, actor_critic, restore_ckpt=None):
#     print(cfg)

#     #  num_agents = sim.actions.shape[0]
#     actor_critic = actor_critic.to(dev)

#     optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

#     # might want to use this
#     # learning_state = LearningState()
#     # if restore_ckpt != None:
#     #     start_update_idx = learning_state.load(restore_ckpt)
#     # else:
#     start_update_idx = 0