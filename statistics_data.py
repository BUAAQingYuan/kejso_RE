__author__ = 'PC-LiNing'

import numpy


# relation label to number
# 0 Cause
# 1 Describe
# 2 From
# 3 Identity
# 4 Medicine
# 5 Part
# 6 Position
# 7 Other
def transfer_label(label):
    if label.startswith('Cause'):
        return 0
    if label.startswith('Describe'):
        return 1
    if label.startswith('From'):
        return 2
    if label.startswith('Identity'):
        return 3
    if label.startswith('Medicine'):
        return 4
    if label.startswith('Part'):
        return 5
    if label.startswith('Position'):
        return 6
    if label.startswith('Other'):
        return 7


def statistic_distribution(labels):
    ids = labels.tolist()
    print("length:"+str(len(ids)))
    counts = [0,0,0,0,0,0,0,0]
    for label in ids:
        counts[int(label)] += 1
    print(counts)


test_label = numpy.load('test_label.npy')
train_label = numpy.load('train_label.npy')

statistic_distribution(test_label)
print("####")
statistic_distribution(train_label)