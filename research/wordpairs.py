import random
from research import NaiveDictLTM

"""
Args:
    targetDistance (int): The number of edges between the target and the prime,
        representing the semantic distance between the target and the prime.
    occControl (double between 0 and 1 inclusive): The co-occurrence frequency of
        the control word.
    occTarget (double between 0 and 1 inclusive): The co-occurrence frequency of
        the target word.
Returns:
     List containing 100 randomized word pairs with frequencies of filler pairs, 
        root/target pairs, and root/control pairs according to occControl and 
        occTarget. 
     Dictionary representing the semantic network by returning the connected nodes
        in the graph.
"""

def wordPairs(targetDistance, occControl, occTarget):
    "Construct semantic network graph"
    semNetwork = {'root': 'dummyR',
                  'root': 'control',
                  'root': 'target',
                  'control': 'dummyC',
                  'filler1': 'dummy1',
                  'filler2': 'dummy2',
                  'filler3': 'dummy3'}
    if targetDistance == 1:
        semNetwork['target'] = 'prime'
    else:
        semNetwork['target'] = 'node1'
        lastNode = 'node1'
        currNode = 'node1'
        i = 1
        for i in range(1, targetDistance):
            currNode = 'node' + str(i)
            semNetwork[lastNode] = currNode
            lastNode = currNode
        semNetwork[currNode] = 'prime'

    "Initialize empty list to store word pairs"
    wordPairList = []
    numTarget = round(occTarget * 100)
    numControl = round(occControl * 100)
    numFiller = 100 - numTarget - numControl
    for i in range(1, numTarget + 1):
        wordPairList.append(['root', 'target'])
    for i in range(1, numControl + 1):
        wordPairList.append(['root', 'control'])
    for i in range(1, numFiller + 1):
        wordPairList.append(['filler1', 'filler2'])

    "Randomize word pair list"
    random.shuffle(wordPairList)

    "Return both the network and word pair list"
    return semNetwork, wordPairList


'''
Args: 
    semNetwork (dictionary): Output of wordPairs function, returns 
        connected nodes in semantic network to be used in the experiment. 
    
Returns: A NaiveDictLTM imported from research -> research -> LTM, that 
    stores the network to be used in the experiment. 
'''

def experimentLTM1(semNetwork):
    network = NaiveDictLTM()
    values = list(semNetwork.values())
    keys = list(semNetwork.keys())
    "Instantiates each key in the dictionary, acting as a node in the graph" \
    "representing the LTM"
    for i in range(len(semNetwork)):
        network.store(keys[i], links_to=values[i])
    return network

def experimentLTM(semNetwork):
    network = NaiveDictLTM()
    for key, value in semNetwork.items():
        network.store(key, links_to=value)
    return network


'''
Args: 
    wordPairList (list): Output of wordpairs function containing 100 randomized word
        pairs with frequencies of filler pairs, root/target pairs, and root/control
        pairs according to occControl and occTarget.
    network (NaiveDictLTM): Output of experimentLTM function LTM that stores the network 
        to be used in the experiment. 

Returns: null
'''
"FIXME Problem with query here"
def experimentActivation(wordPairList, network):
    for i in range(len(wordPairList)):
        network.retrieve(wordPairList[i][0])
        network.retrieve(wordPairList[i][1])



testNetwork = wordPairs(3, 0.2, 0.2)
testLTM = experimentLTM(testNetwork[0])
experimentActivation(testNetwork[1], testLTM)
