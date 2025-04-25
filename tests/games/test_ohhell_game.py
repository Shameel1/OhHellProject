import unittest 
import numpy as np 
import random

from rlohhell.games.ohhell.game import OhHellGame as Game
from rlohhell.games.ohhell.player import OhHellPlayer as Player
from rlohhell.games.ohhell.round import OhHellRound as Round
from rlohhell.games.ohhell.judger import OhHellJudger as Judger
from rlohhell.games.ohhell.utils import ACTION_LIST, determine_winner, int2rank
from rlohhell.games.base import Card

class TestOhHellMethods(unittest.TestCase):

    def test_get_num_actions(self):
        game = Game()
        num_players = game.get_num_players()
        self.assertEqual(num_players, 4)

    def test_get_num_actions(self):
        game = Game()
        num_actions = game.get_num_actions()
        self.assertEqual(num_actions, 63)

    def test_init_game(self):
        game = Game()
        state, _ = game.init_game()
        game.step(1)
        game.step(1)
        game.step(2)
        actions = game.get_legal_actions()
        self.assertNotIn(6, actions)
        
    def test_step(self):
        game = Game()

        # bid
        game.init_game()
        init_tricks = np.array(game.round.proposed_tricks)
        game.step(1)
        game.step(1)
        game.step(1)
        game.step(1)
        proposed_tricks = game.round.proposed_tricks
        init_tricks += 1
        init_tricks = list(init_tricks)
        self.assertListEqual(init_tricks, proposed_tricks)
        action = random.choice(game.get_legal_actions())
        state, next_player_id = game.step(action)
        current = game.round.current_player
        self.assertEqual(len(state['played_cards']), 1)
        self.assertEqual(next_player_id, current)

    def test_get_payoffs(self):
        game = Game()
        game.init_game()
        while not game.is_over():
            actions = game.get_legal_actions()
            action = random.choice(actions)
            state, _ = game.step(action)
        payoffs = game.get_payoffs()
        proposed_tricks = game.round.proposed_tricks
        tricks_won = [player.tricks_won for player in game.players]
        expected_payoff = [ tricks+10 if tricks == proposed_tricks[k] else tricks for k, tricks in enumerate(tricks_won) ]
        self.assertListEqual(expected_payoff, list(payoffs))

    def test_step_back(self):
        game = Game(allow_step_back=True)
        _, player_id = game.init_game()
        action = random.choice(game.get_legal_actions())
        game.step(action)
        game.step_back()
        self.assertEqual(game.round.current_player, player_id)
        self.assertEqual(len(game.history), 0)
        success = game.step_back()
        self.assertEqual(success, False)  

    def test_determine_winner(self):
        trump_card = Card('D', 'T')
        played_cards = [Card('D','A'), Card('S','2'), Card('D','3'), Card('H','4')]
        winner = determine_winner(played_cards, trump_card)
        self.assertEqual(winner, 0)

    def test_player_get_player_id(self):
        player = Player(0, np.random.RandomState())
        self.assertEqual(0, player.get_player_id())

    def test_previously_played_cards(self):
        game = Game()
        game.init_game()
        while not game.is_over():
            actions = game.get_legal_actions()
            action = random.choice(actions)
            state, _ = game.step(action)
        num_played_cards = len(game.previously_played_cards)
        num_played_cards_player3 = len(game.players[2].played_cards)
        self.assertEqual(num_played_cards, 40)
        self.assertEqual(num_played_cards_player3, 10)  
    
if __name__ == '__main__':
    unittest.main() 


# The below code is for running the game in the shell
# Import rlohhell with 'pip install -e .' while in the folder with setup.py
# type 'python'
# Then run the below code

from rlohhell.games.ohhell.player import OhHellPlayer as Player
from rlohhell.games.ohhell.game import OhHellGame as Game
from rlohhell.games.base import Card

# Create function to get number of wins.

game = Game()
game.init_game()

def get_legal():
    legal_actions = game.get_legal_actions()
    if isinstance(legal_actions[0], int):
        print("Bid tricks:    ",legal_actions)
    else:
        visible_legal = [card.get_index() for card in legal_actions]
        print("Legal Options: ",visible_legal)

def get_hand():
    my_cards = game.players[game.current_player].hand
    visible_hand = [card.get_index() for card in my_cards]
    print("Current Hand:  ", visible_hand)
    
def get_trump():
    trump = game.trump_card
    trump_v = trump.get_index()
    print("Trump Card:     [",trump_v,"]")
   
# Quick rain-check
def status():
    print("\n")
    get_hand()
    get_legal()
    get_trump()
    print("\n")
    
# Step functions 
# Bidding number of tricks
def step(x):
    game.step(x)
    
# Playing a card
def step_c(x, y):
    game.step(Card(x,y))
    
