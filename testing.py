import rlcard
from rlcard.agents import RandomAgent, DQNAgent
import pprint
pp = pprint.PrettyPrinter(depth=4)
import sys
import os
from rlcard.envs.registration import register

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to sys.path
sys.path.append(current_dir)

register(
    env_id='ohhell',
    entry_point='rlohhell.envs.ohhell:OhHellEnv',  # NOT a file path
)

env = rlcard.make('ohhell')

trainable_agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64,64],
)
agents = [RandomAgent(num_actions=env.num_actions) for i in range (env.num_players-1)]
agents.append(trainable_agent)
env.set_agents(agents)

obs, _ = env.reset()
pp.pprint(obs)

trajectories, payoffs = env.run()
for i, episode in enumerate(trajectories[0]):
    if i%2 == 0:
        print(f"State {i}:")
        pp.pprint(episode)
    
print(payoffs) # [0.0]