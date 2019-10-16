import csv
import numpy as np
import random
import ast
import argparse

parser = argparse.ArgumentParser(description='eval metaphor detection')
parser.add_argument('--pred',  help='pred file', default='None')
args = parser.parse_args()

gold_labels = []
with open('./data_release/val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        label_seq = ast.literal_eval(line[2])
        words = line[0].split()
        for i in range(len(words)):
            gold_labels.append(label_seq[i])

predictions = []
with open(args.pred) as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pred = ast.literal_eval(line[1])
        predictions.append(pred)

# # set to all 1
# for i in range(len(predictions)):
#     predictions[i] = 1

# # set to random (0,1)
# for i in range(len(predictions)):
#     predictions[i] = random.randrange(2)

# evaluate
assert(len(predictions) == len(gold_labels))
total_examples = len(predictions)


num_correct = 0
confusion_matrix = np.zeros((2, 2))
for i in range(total_examples):
    if predictions[i] == gold_labels[i]:
        num_correct += 1
    confusion_matrix[predictions[i], gold_labels[i]] += 1

assert(num_correct == confusion_matrix[0, 0] + confusion_matrix[1, 1])
accuracy = 100 * num_correct / total_examples
precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
met_f1 = 2 * precision * recall / (precision + recall)


print('P, R, F1, Acc.')
print(precision, recall, met_f1, accuracy)