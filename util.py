#! /usr/bin/env python
# -*- coding:utf-8 -*-


import traceback
from deuces import Card
from deuces import Evaluator
from deuces import Deck
from itertools import combinations
import numpy as np



def printtolog(msg):
    with open('log.txt', 'a') as fw:
        fw.write( '{0}\n'.format( msg) )


class DataParser:
    def __init__(self, transactions):
        self.transactions = transactions
        self.action_to_int = {
            'fold':1,
            'check':2,
            'bet':3,
            'call':4,
            'raise':5,
            'allin':6,
            'other':7
        }
        self.roundname_to_int = {
            'Deal':1,
            'Flop':2,
            'Turn':3,
            'River':4,
            'other':5       
        }
        self.fea_len_current = 29 # current feature length
        self.score_min = 10000# deuces evaluate score, the small, the better
        # re-arrange player number
        player_num_dict = {}
        num=1
        for t in self.transactions:
            player = t['player']
            if(player not in player_num_dict):
                player_num_dict[player] = num
                num+=1
        for i in range(len(self.transactions)):
            player = self.transactions[i]['player']
            player_num = player_num_dict[player]
            self.transactions[i]['observation']['player_num'] = player_num
            
            
    def map_cards(self, card):
        card = card.encode('ascii','ignore')
        color = {"h":0, "s":1, "c":2, "d":3}
        number = {"t":10, "j":11, "q":12, "k":13, "a":1, "T":10, "J":11, "Q":12, "K":13, "A":1}
        for i in range(2, 10):
            number[str(i)] = i
        return (int(number[card[0]]) - 1) * 4 + color[card[1].lower()] + 1
    
    
    def get_score_by_simulate(self, board_cards, hand_cards, iteration=5):
        
        try:
            score_min = self.score_min
            board = []
            for x in board_cards:
                c = Card.new('{0}{1}'.format(x[0], x[1].lower()))
                board.append(c)
            
            hand = []
            for x in hand_cards:
                c = Card.new('{0}{1}'.format(x[0], x[1].lower()))
                hand.append(c)
            if(len(hand)+len(board)<5):
                score_list = []
                for i in range(iteration):
                    evaluator = Evaluator()
                    deck = Deck()
                    random_cards = deck.draw(5-len(board)-len(hand))
                    if(isinstance(random_cards,int)):
                        random_cards = [random_cards]
                    score = evaluator.evaluate(board+random_cards, hand)
                    score_list.append(score)
                    
                score_list.remove(max(score_list))    
                score_list.remove(min(score_list))    
                
                return sum(score_list)/float(len(score_list))
            else:
                
                for board_five_match in combinations(board,5-len(hand)):
                    evaluator = Evaluator()
                    score = evaluator.evaluate(tuple(board_five_match), tuple(hand))
                    if(score<score_min):
                        score_min=score
                return score_min
        except Exception as e:
            traceback.print_exc()
            printtolog('EXCEPTION={0}'.format(e))
            return score_min
        
    def get_fea(self, trans, trans_index):
        try:
            fea = []

            if(trans['action'] in self.action_to_int):
                action_int = self.action_to_int[trans['action']]
            else:
                action_int = self.action_to_int['other']

            if(trans['observation']['roundName'] in self.roundname_to_int):
                round_int = self.roundname_to_int[trans['observation']['roundName']]
            else:
                round_int = self.roundname_to_int['other']

            if(trans['observation']['is_big_blind']):
                big_blind_flag = 1
            else:
                big_blind_flag = 0

            if(trans['observation']['is_small_blind']):
                small_blind_flag = 1
            else:
                small_blind_flag = 0

            # 20 features here
            fea.append(action_int)
            fea.append(round_int)
            fea.append(big_blind_flag)
            fea.append(small_blind_flag)
            fea.append(trans['observation']['player_num'])
            fea.append(trans['observation']['chips'])
            fea.append(trans['observation']['amount'])
            fea.append(trans['observation']['minBet'])
            fea.append(trans['observation']['totalBet'])
            fea.append(trans['observation']['betCount'])
            fea.append(trans['observation']['initChips'])
            fea.append(trans['observation']['big_blind_amount'])
            fea.append(trans['observation']['small_blind_amount'])
            # board_cards
            board_cards = []
            for i in range(5):
                if(i<len(trans['observation']['board_cards'])):
                    c = trans['observation']['board_cards'][i]
                    board_cards.append(self.map_cards(c))
                else:
                    board_cards.append(0)
            fea+=sorted(board_cards)        
            # self_cards
            self_cards = []
            for i in range(2):
                if(i<len(trans['observation']['self_cards'])):
                    c = trans['observation']['self_cards'][i]
                    self_cards.append(self.map_cards(c))
                else:
                    self_cards.append(0)
            fea+=sorted(self_cards)
            # card score
            card_score = self.get_score_by_simulate(trans['observation']['board_cards'], trans['observation']['self_cards'])
            # card contribution
            contribution = self.score_min
            if(len(trans['observation']['board_cards'])>2):
                card_score_self = self.get_score_by_simulate(trans['observation']['board_cards'], trans['observation']['self_cards'])
                card_score_noself = self.get_score_by_simulate(trans['observation']['board_cards'], [])
                contribution = card_score_noself-card_score_self
            # 2 features added here
            fea.append(card_score)
            fea.append(contribution)
            # other player action statistic fea added here
            # find this round player {start, this round}
            p = trans['player']
            start_p = 0
            end_p = trans_index
            len_transactions = len(self.transactions)
            if(len_transactions==0 or trans_index==0):
                for i in self.action_to_int:
                    fea.append(0)
            else:        
                for i in range(trans_index):
                    j = trans_index-i-1
                    trans_here = self.transactions[j]
                    
                    if(trans['player']==p):
                        start_p = j
                        break
                action_percent = [0]*len(self.action_to_int)
                # get action count
                for i in range(end_p-start_p):
                    j = start_p+1+i
                    trans = self.transactions[j]
                    idx = self.action_to_int[trans['action']]-1
                    action_percent[idx]+=1
                # get action percent 
                for i in range(len(action_percent)):
                    action_percent[i] = float(action_percent[i])/float(len(action_percent))
                fea += action_percent
            
            
            return fea
        except Exception as e:
            traceback.print_exc()
            printtolog('EXCEPTION={0}'.format(e))
            return []


    def get_reward(self, trans):
        return trans['reward']
        
    def get_observations(self):
        j = len(self.transactions)-1
        if(j<=0):
            return [0]*self.fea_len_current
        else:
            trans = self.transactions[j]
            fea = self.get_fea(trans,j)
            return fea
        
    def get_dataset(self):
        x_data = []
        y_data = []
        for i in range(len(self.transactions)):
            trans = self.transactions[i]
            x_data.append(self.get_fea(trans, i))
            y_data.append(self.get_reward(trans))
        return x_data,y_data

def test():
    TBL_TRANS=[{'action': u'call', 'player': u'209ba76313536d3398d22e5254d6a296', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'TC', u'JS'], 'betCount': 0, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 20, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 1, 'totalBet': 50, 'chips': 980, 'minBet': 0}}, {'action': u'call', 'player': u'2ff5bf55f618e9fb9fd205f190d57f29', 'reward': 469.0, 'event': u'__show_action', 'observation': {'self_cards': [u'AH', u'3H'], 'betCount': 0, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 20, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 2, 'totalBet': 70, 'chips': 980, 'minBet': 0}}, {'action': u'call', 'player': u'64d95868195234fa73983294f9e1e38a', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'QC', u'2D'], 'betCount': 0, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 20, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 3, 'totalBet': 90, 'chips': 980, 'minBet': 0}}, {'action': u'bet', 'player': u'd1c7e83c0d06517bd2c91f469b7be4b2', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'5D', u'JC'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 118, 'is_big_blind': False, 'is_small_blind': True, 'player_num': 4, 'totalBet': 208, 'chips': 872, 'minBet': 0}}, {'action': u'call', 'player': u'd4a9e2cc9c74a077635630c6feaddeaa', 'reward': 0, 'event': u'__action', 'observation': {'self_cards': [u'5S', u'9C'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 0, 'is_big_blind': True, 'is_small_blind': False, 'player_num': 5, 'totalBet': 208, 'chips': 980, 'minBet': 108}}, {'action': u'call', 'player': u'd4a9e2cc9c74a077635630c6feaddeaa', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'5S', u'9C'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 108, 'is_big_blind': True, 'is_small_blind': False, 'player_num': 6, 'totalBet': 316, 'chips': 872, 'minBet': 0}}, {'action': u'call', 'player': u'209ba76313536d3398d22e5254d6a296', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'TC', u'JS'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 108, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 7, 'totalBet': 424, 'chips': 872, 'minBet': 0}}, {'action': u'call', 'player': u'2ff5bf55f618e9fb9fd205f190d57f29', 'reward': 469.0, 'event': u'__show_action', 'observation': {'self_cards': [u'AH', u'3H'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 108, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 8, 'totalBet': 532, 'chips': 872, 'minBet': 0}}, {'action': u'call', 'player': u'64d95868195234fa73983294f9e1e38a', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'QC', u'2D'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [], 'roundName': u'Deal', 'amount': 108, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 9, 'totalBet': 640, 'chips': 872, 'minBet': 0}}, {'action': u'bet', 'player': u'd1c7e83c0d06517bd2c91f469b7be4b2', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'5D', u'JC'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 29, 'is_big_blind': False, 'is_small_blind': True, 'player_num': 10, 'totalBet': 669, 'chips': 843, 'minBet': 0}}, {'action': u'fold', 'player': u'd4a9e2cc9c74a077635630c6feaddeaa', 'reward': 0, 'event': u'__action', 'observation': {'self_cards': [u'5S', u'9C'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 0, 'is_big_blind': True, 'is_small_blind': False, 'player_num': 0, 'totalBet': 669, 'chips': 872, 'minBet': 29}}, {'action': u'fold', 'player': u'd4a9e2cc9c74a077635630c6feaddeaa', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'5S', u'9C'], 'betCount': 1, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 0, 'is_big_blind': True, 'is_small_blind': False, 'player_num': 0, 'totalBet': 669, 'chips': 872, 'minBet': 0}}, {'action': u'bet', 'player': u'209ba76313536d3398d22e5254d6a296', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'TC', u'JS'], 'betCount': 2, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 50, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 0, 'totalBet': 719, 'chips': 822, 'minBet': 0}}, {'action': u'bet', 'player': u'2ff5bf55f618e9fb9fd205f190d57f29', 'reward': 469.0, 'event': u'__show_action', 'observation': {'self_cards': [u'AH', u'3H'], 'betCount': 3, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 688, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 0, 'totalBet': 1407, 'chips': 184, 'minBet': 0}}, {'action': u'fold', 'player': u'64d95868195234fa73983294f9e1e38a', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'QC', u'2D'], 'betCount': 3, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 0, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 0, 'totalBet': 1407, 'chips': 872, 'minBet': 0}}, {'action': u'fold', 'player': u'd1c7e83c0d06517bd2c91f469b7be4b2', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'5D', u'JC'], 'betCount': 3, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 0, 'is_big_blind': False, 'is_small_blind': True, 'player_num': 0, 'totalBet': 1407, 'chips': 843, 'minBet': 0}}, {'action': u'fold', 'player': u'209ba76313536d3398d22e5254d6a296', 'reward': 0, 'event': u'__show_action', 'observation': {'self_cards': [u'TC', u'JS'], 'betCount': 3, 'big_blind_amount': 20, 'self_cards_contribution': 0, 'initChips': 1000, 'small_blind_amount': 10, 'board_cards': [u'3S', u'KD', u'3D'], 'roundName': u'Flop', 'amount': 0, 'is_big_blind': False, 'is_small_blind': False, 'player_num': 0, 'totalBet': 1407, 'chips': 822, 'minBet': 0}}]

    dp = DataParser(TBL_TRANS)
    ob = dp.get_observations()
    x_data,y_data = dp.get_dataset()
    print('observation len={0}'.format(len(ob)))
    print('observation:')
    print(ob)
    print('x_data={0}'.format(x_data))
    print('y_data={0}'.format(y_data))
    print('x_data.shape={0}'.format(np.array(x_data).shape))
    print('y_data.shape={0}'.format(np.array(y_data).shape))
    
    for x in x_data:
        print('x_data_len={0}'.format(len(x)))
    
    
    
if __name__=='__main__':
    test()