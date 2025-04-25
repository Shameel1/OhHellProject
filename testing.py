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
from rlohhell.games.ohhell.utils import ACTION_LIST

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
print('TRUMP:', obs['trump']) #
trajectories, payoffs = env.run(is_training=True)
# Iterate through the trajectories to print the whole game
for step in range(len(trajectories[0]) // 2):  # Each step has states and actions
    print(f"Step {step}:")
    for player_id in range(env.num_players):
        state = trajectories[player_id][step * 2]  # State at this step
        action = trajectories[player_id][step * 2 + 1]  # Action at this step
        print(f"\t Player {player_id}, Legal Moves: {state['raw_legal_actions']}, Action: {ACTION_LIST[action]}")
    print("result:", state['players_tricks_won'])
print(payoffs) # [0.0]