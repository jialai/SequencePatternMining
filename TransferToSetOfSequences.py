import csv
import pandas as pd
import numpy as np
import csv
import copy
import itertools
from collections import defaultdict
from operator import itemgetter
#思路：一开始想着用循环之类的去手动分，后来发现有groupby这种可以直接分，然后用循环转换成list
#这个方法太笨了，而且不起什么作用
# sequences = []
# sessionList = []
# tempSession='8RS7WBtgvy'
# sort(data[3])
# for line in csv.reader((row for row in open("mobsos_MONITORING.csv")), delimiter='\t'):
#     if len(line) == 0:
#         continue
#     lineList=line[0].split(',')
#     l=len(lineList)
#     if l<17 or lineList[16] == '':
#         continue
#     if tempSession==lineList[3]:
#         sessionList.append(lineList[16])
#         print(tempSession)
#     else:
#         tempSession=lineList[3]
#         sequences.append(sessionList[:])
#         sessionList.clear()
#
# print(sequences)
def projectSequence(sequence, prefix, newEvent):
    result = None
    for i, itemset in enumerate(sequence):
        if result is None:
            if (not newEvent) or i > 0:
                if (all(x in itemset for x in prefix)):
                    result = [list(itemset)]
        else:
            result.append(copy.copy(itemset))
    return result

"""
Projects a dataset according to a given prefix, as done in PrefixSpan

Args:
    dataset: the dataset the projection is built from
    prefix: the prefix that is searched for in the sequence
    newEvent: if set to True, the first itemset is ignored
Returns:
    A (potentially empty) list of sequences
"""
def projectDatabase(dataset, prefix, newEvent):
    projectedDB = []
    for sequence in dataset:
        seqProjected = projectSequence(sequence, prefix, newEvent)
        if not seqProjected is None:
            projectedDB.append(seqProjected)
    return projectedDB
"""
Generates a list of all items that are contained in a dataset
"""
def generateItems(dataset):
    return sorted(set ([item for sublist1 in dataset for sublist2 in sublist1 for item in sublist2]))

"""
Computes a defaultdict that maps each item in the dataset to its support
"""
def generateItemSupports(dataset, ignoreFirstEvent=False, prefix=[]):
    result = defaultdict(int)
    for sequence in dataset:
        if ignoreFirstEvent:
            sequence = sequence[1:]
        cooccurringItems = set()
        for itemset in sequence:
            if all(x in itemset for x in prefix):
                for item in itemset:
                    if not item in prefix:
                        cooccurringItems.add(item)
        for item in cooccurringItems:
            result [item] += 1
    return sorted(result.items())


"""
The PrefixSpan algorithm. Computes the frequent sequences in a seqeunce dataset for a given minSupport

Args:
    dataset: A list of sequences, for which the frequent (sub-)sequences are computed
    minSupport: The minimum support that makes a sequence frequent
Returns:
    A list of tuples (s, c), where s is a frequent sequence, and c is the count for that sequence
"""


def prefixSpan(dataset, minSupport):
    result = []
    itemCounts = generateItemSupports(dataset)
    for item, count in itemCounts:
        if count >= minSupport:
            newPrefix = [[item]]
            result.append((newPrefix, count))
            result.extend(prefixSpanInternal(projectDatabase(dataset, [item], False), minSupport, newPrefix))
    return result


def prefixSpanInternal(dataset, minSupport, prevPrefixes=[]):
    result = []

    # Add a new item to the last element (==same time)
    itemCountSameEvent = generateItemSupports(dataset, False, prefix=prevPrefixes[-1])
    for item, count in itemCountSameEvent:
        if (count >= minSupport) and item > prevPrefixes[-1][-1]:
            newPrefix = copy.deepcopy(prevPrefixes)
            newPrefix[-1].append(item)
            result.append((newPrefix, count))
            result.extend(prefixSpanInternal(projectDatabase(dataset, newPrefix[-1], False), minSupport, newPrefix))

    # Add a new event to the prefix
    itemCountSubsequentEvents = generateItemSupports(dataset, True)
    for item, count in itemCountSubsequentEvents:
        if count >= minSupport:
            newPrefix = copy.deepcopy(prevPrefixes)
            newPrefix.append([item])
            result.append((newPrefix, count))
            result.extend(prefixSpanInternal(projectDatabase(dataset, [item], True), minSupport, newPrefix))
    return result
def groupData(df):
    seq = []
    obj=df['METHOD_NAME'].groupby(df['SESSION_ID'])
    for id, group in obj:
        # print(id)
        # print(list(group))
        seq.append(list(group))
    # print(seq)
    return seq

data=pd.read_csv("mobsos_MONITORING.csv",na_values='NULL')
dataForAnalysis=data[['SESSION_ID','METHOD_NAME']]
noNnaDataForAnalysis=dataForAnalysis.dropna(axis = 0)
listOfData=groupData(noNnaDataForAnalysis)
result=prefixSpan (listOfData, 1000)
print(result)


