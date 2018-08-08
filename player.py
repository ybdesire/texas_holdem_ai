#! /usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.externals import joblib
from deuces import Card
from deuces import Evaluator
from deuces import Deck
from itertools import combinations
import time
import json
from websocket import create_connection
import copy
import traceback
import zmq
from util import DataParser



def printtolog(msg):
    print(msg)
    with open('log.txt', 'a') as fw:
        fw.write( '{0}\n'.format( msg) )


# global
model = joblib.load('ai2.0_model_1.jl')
print('model ready')
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5555')
        
# global 
PLAYER_COUNT_TOTAL = 10 # 10 player
player_count = 0
self_reward_calc = False
# global transaction table
TBL_TRANS = [] # each players action and boservation
TBL_FMT = {# table format, should be deep copy (d2 = copy.deepcopy(d))
    'event':'',
    'player':'',
    'observation':
    {
        'is_big_blind':False,
        'is_small_blind':False,
        'player_num':0,
        'chips':0,
        'amount':0,
        'minBet':0,
        
        'roundName':'',
        'totalBet':0,
        'betCount':0,
        'initChips':0,
        'big_blind_amount':0,
        'small_blind_amount':0,
        'board_cards':[],
        'self_cards':[],
        'self_cards_contribution':0
    },
    'action':'',
    'reward':0
}
PLAYERS = {}# key=player name, value=player num
SELF_TRANS = []# latest self transaction


def test_fea():
    global player_count,TBL_FMT,PLAYER_COUNT_TOTAL,TBL_TRANS,PLAYERS
    if(len(TBL_TRANS)>1):
        dp = DataParser(TBL_TRANS)
        ob = dp.get_observations()
        x_data,y_data = dp.get_dataset()
        printtolog('observation len={0}'.format(len(ob)))
        printtolog('observation:')
        printtolog(ob)
        printtolog('x_data={0}'.format(x_data))
        printtolog('y_data={0}'.format(y_data))
        printtolog('TBL_TRANS')
        printtolog(TBL_TRANS)

# send reward to server side (dqn)
def update_reward():
    global model,socket
    dp = DataParser(TBL_TRANS)
    fea = dp.get_observations()
    reward = model.predict(fea)
    observation_ = fea
    socket.send(str({'reward':reward, 'observation_':observation_}))
    msg = socket.recv()
    
# send training msg to server side (dqn)
def train_dqn():
    global socket
    socket.send(str({'training':''}))
    msg = socket.recv()

        
def add_transcation(event_name, data):
    try:
        global player_count,TBL_FMT,PLAYER_COUNT_TOTAL,TBL_TRANS,PLAYERS,SELF_TRANS,self_reward_calc
        
        if('__new_peer'==event_name):# new player adding the game
            for player in data:
                if(player not in PLAYERS):
                    PLAYERS[player]=0# add one player with default num 0
        if('__left'==event_name):# player leaving the game
            tmp_players = []
            for player in data:
                if(player not in PLAYERS):
                    tmp_players[player]=PLAYERS[player]
            PLAYERS = tmp_players
        if('__round_end'==event_name):# round end, update hand card and calculate reward
            player_dict = {}
            for p in data['players']:
                if(p['playerName'] not in player_dict):
                    player_dict[p['playerName']] = {}
                player_dict[p['playerName']]['self_cards'] = p['cards']
                player_dict[p['playerName']]['winMoney'] = p['winMoney']
            # action statistic and self_cards update    
            player_action_count_dict = {}    
            for transaction in  TBL_TRANS:
                p = transaction['player']
                if(transaction['observation']['self_cards']==[]):
                    transaction['observation']['self_cards'] = player_dict[p]['self_cards']
                a = transaction['action']
                
                if(p not in player_action_count_dict):
                    player_action_count_dict[p]={}
                    player_action_count_dict[p]['action_count_nonfold']=0# total action count except fold
                    player_action_count_dict[p]['action_count']=0
                if(a not in player_action_count_dict[p]):
                    player_action_count_dict[p][a]=0
                player_action_count_dict[p][a]+=1
                if(a!='fold'):
                    player_action_count_dict[p]['action_count_nonfold']+=1
                player_action_count_dict[p]['action_count']+=1
                
            # calc reward    
            for transaction in  TBL_TRANS:
                p = transaction['player']
                a = transaction['action']
                winMoney = player_dict[p]['winMoney']
                if(a=='fold'):
                    transaction['reward'] = 0
                if(winMoney==0):
                    transaction['reward'] = 0
                else:
                    if(player_action_count_dict[p]['action_count_nonfold']!=0):
                        transaction['reward'] = float(winMoney)/player_action_count_dict[p]['action_count_nonfold']
                    elif(player_action_count_dict[p]['action_count']!=0):
                        transaction['reward'] = float(winMoney)/player_action_count_dict[p]['action_count']
                    else:
                        transaction['reward'] = float(winMoney)
            printtolog('round-{0}, TBL_TRANS={1}'.format(data['table']['roundCount'], TBL_TRANS))  
            
            # CLEAN each round
            TBL_TRANS = []
            SELF_TRANS = []
            # train dqn 
            train_dqn()
        if('__show_action'==event_name):# other player has action
            player_count+=1
            transaction = copy.deepcopy(TBL_FMT)
            transaction['event'] = event_name
            transaction['player'] = data['action']['playerName']
            transaction['action'] = data['action']['action']
            if(transaction['player']==data['table']['bigBlind']['playerName']):
                transaction['observation']['is_big_blind'] = True
            if(transaction['player']==data['table']['smallBlind']['playerName']):
                transaction['observation']['is_small_blind'] = True
            if(transaction['observation']['player_num']==0):
                if(player_count<=PLAYER_COUNT_TOTAL):
                    transaction['observation']['player_num']=player_count
            transaction['observation']['chips'] = data['action']['chips']
            if(u'amount' in data['action']):
                transaction['observation']['amount'] = data['action']['amount']
            # data from table 
            transaction['observation']['roundName'] = data['table']['roundName']
            transaction['observation']['totalBet'] = data['table']['totalBet']
            transaction['observation']['betCount'] = data['table']['betCount']
            transaction['observation']['initChips'] = data['table']['initChips']
            transaction['observation']['big_blind_amount'] = data['table']['bigBlind']['amount']
            transaction['observation']['small_blind_amount'] = data['table']['smallBlind']['amount']
            transaction['observation']['board_cards'] = data['table']['board']
            for p in data['players']:
                if(transaction['player']==p['playerName']):
                    if('cards' in p):
                        transaction['observation']['self_cards'] = p['cards']
            TBL_TRANS.append(transaction)
            
            if(self_reward_calc):
                update_reward()
                self_reward_calc = False
                for i in range(len(TBL_TRANS)):
                    j = len(TBL_TRANS) - i - 1
                    if(j>=0):
                        trans = TBL_TRANS[j]
                        if(trans['player']==SELF_TRANS['player']):
                            SELF_TRANS = trans
                            break
            
        if('_action'==event_name):# get self transaction without action
            player_count+=1
            transaction = copy.deepcopy(TBL_FMT)
            transaction['event'] = event_name
            transaction['player'] = data['self']['playerName']
            transaction['action'] = ''
            if(transaction['player']==data['game']['bigBlind']['playerName']):
                transaction['observation']['is_big_blind'] = True
            if(transaction['player']==data['game']['smallBlind']['playerName']):
                transaction['observation']['is_small_blind'] = True
            if(transaction['observation']['player_num']==0):
                if(player_count<=PLAYER_COUNT_TOTAL):
                    transaction['observation']['player_num']=player_count
            transaction['observation']['chips'] = data['self']['chips']
            if(u'amount' in data['self']):
                transaction['observation']['amount'] = data['self']['amount']
            # data from table 
            transaction['observation']['roundName'] = data['game']['roundName']
            transaction['observation']['totalBet'] = data['game']['totalBet']
            transaction['observation']['betCount'] = data['game']['betCount']
            transaction['observation']['big_blind_amount'] = data['game']['bigBlind']['amount']
            transaction['observation']['small_blind_amount'] = data['game']['smallBlind']['amount']
            transaction['observation']['board_cards'] = data['game']['board']
            transaction['observation']['self_cards'] = data['self']['cards']
            SELF_TRANS = transaction
            self_reward_calc=True
            
        return True
    except Exception as e:
        traceback.print_exc()
        printtolog('EXCEPTION={0}'.format(e))
        return False

        

def get_score_by_simulate(board_cards, hand_cards, iteration=5):
    
    try:
        score_min = 10000# deuces evaluate score, the small, the better
        board = []
        for x in board_cards:
            x = x.encode('ascii','ignore')
            c = Card.new('{0}{1}'.format(x[0], x[1].lower()))
            board.append(c)
        
        hand = []
        for x in hand_cards:
            x = x.encode('ascii','ignore')
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


def get_score(hand_cards, board_cards):

    board = []
    for x in board_cards:
        x = x.encode('ascii','ignore')
        c = Card.new('{0}{1}'.format(x[0], x[1].lower()))
        board.append(c)
    
    hand = []
    for x in hand_cards:
        x = x.encode('ascii','ignore')
        c = Card.new('{0}{1}'.format(x[0], x[1].lower()))
        hand.append(c)


    score_min = 9999999
    if(len(hand)+len(board)<=5):
        return score_min
    else:
        
        for board_five_match in combinations(board,5-len(hand)):
            evaluator = Evaluator()
            score = evaluator.evaluate(tuple(board_five_match), tuple(hand))
            if(score<score_min):
                score_min=score
        return score_min


def makeActionStr(actionNum, amount):
    action = {
            "eventName" : "__action",
            "data" : {
                "action" : "xxxx"
            }
        }
    if actionNum == 1:
        action["data"]["action"] = "fold"
    elif actionNum == 2:
        action["data"]["action"] = "check"
    elif actionNum == 3:
        action["data"]["action"] = "bet"
        action["data"]["amount"] = amount
    elif actionNum == 4:
        action["data"]["action"] = "call"
    elif actionNum == 5:
        action["data"]["action"] = "raise"
        action["data"]["amount"] = amount
    elif actionNum == 6:
        action["data"]["action"] = "allin"
    else:
        action["data"]["action"] = "check"
    return json.dumps(action)


def aiAction(action, data):
    global socket
    try:
        if action == "__action" or action == "__bet":
            dp = DataParser(TBL_TRANS+SELF_TRANS)
            observation = dp.get_observations()
            # send observation to server and get action
            socket.send(str({'observation':observation}))
            msg = socket.recv()
            actionNum = int(msg)
            action_json = makeActionStr(actionNum, amount=int(data['self']['minBet']))
            return action_json
    except Exception as e:
        traceback.print_exc()
        printtolog('EXCEPTION={0}'.format(e))
        return makeActionStr(2,0)# check



# pip install websocket-client
ws = ""

def takeAction(decision):

    ws.send(decision)



def doListen():
    try:
        global ws
        ws = create_connection("ws://10.0.0.0")# official web socket server
        ws.send(json.dumps({
            "eventName": "__join",
            "data": {
                "playerName": "xxxxxxx"#official
            }
        }))
        while 1:
            result = ws.recv()
            msg = json.loads(result)
            event_name = msg["eventName"]
            data = msg["data"]
            add_transcation(event_name, data)    
            decision = aiAction(event_name, data)
            if decision:
                takeAction(decision)
    except Exception as e:
        import traceback
        traceback.print_exc()
        doListen()


if __name__ == '__main__':
    doListen()
