import csv
import numpy as np
import ast
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from p2_model1 import labelsCount, transitionProbability

# process file
# format: sentence, pos_seq, label_seq
def getCorpus(filename):
    corpus = []
    posSeq = []
    labelSeq = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            # define the start word <s>
            words = line[0].split()
            newWordLine = ['<s>']
            for i in range(len(words)):
                newWordLine.append(words[i])
            corpus.append(newWordLine)

            # define the start pos tag
            pos = ast.literal_eval(line[1])
            newPosLine = ['None']
            for i in range(len(pos)):
                newPosLine.append(pos[i])
            posSeq.append(newPosLine)

            # define the label start symbol
            labels = ast.literal_eval(line[2])
            newLabelLine = [100]
            for i in range(len(labels)):
                newLabelLine.append(labels[i])
            labelSeq.append(newLabelLine)
    return corpus, posSeq, labelSeq

def getTestCorpus(filename):
    corpus = []
    posSeq = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            # define the start word <s>
            words = line[0].split()
            newWordLine = ['<s>']
            for i in range(len(words)):
                newWordLine.append(words[i])
            corpus.append(newWordLine)

            # define the start pos tag
            pos = ast.literal_eval(line[1])
            newPosLine = ['None']
            for i in range(len(pos)):
                newPosLine.append(pos[i])
            posSeq.append(newPosLine)
    return corpus, posSeq

""" features created
        1. current word
        2. current pos tag
        3. previous pos tag
        4. following pos tag
"""
def createFeatures(corpus, posList):
    ## feature: previous word's pos tag
    prev_pos = copy.deepcopy(posList)
    ## feature: next word's pos tag
    next_pos = copy.deepcopy(posList)
    ## merge features into list of dicts
    X_features = list()
    for i, sentence in enumerate(corpus):
        prev_pos[i].insert(0, "None")
        prev_pos[i].pop()
        next_pos[i].pop(0)
        next_pos[i].append("None")
        for j, word in enumerate(sentence):
            feature = dict()
            feature["word"] = word
            feature["pos"] = posList[i][j]
            feature["prev_pos"] = prev_pos[i][j]
            feature["next_pos"] = next_pos[i][j]
            X_features.append(feature)
    return X_features

def createFeaturesForLine(line, posList):
    ## feature: previous word's pos tag
    prev_pos = posList.copy()
    prev_pos.insert(0, "None")
    prev_pos.pop()
    ## feature: next word's pos tag
    next_pos = posList.copy()
    next_pos.pop(0)
    next_pos.append("None")
    ## merge features into list
    X_features = list()
    for i, word in enumerate(line):
        feature = dict()
        feature["word"] = word
        feature["pos"] = posList[i]
        feature["prev_pos"] = prev_pos[i]
        feature["next_pos"] = next_pos[i]
        X_features.append(feature)
    return X_features

# bigram viterbi
def viterbi(corpus, posList, clf, vector):
    possibleLabels = [0, 1]
    output = []
    for i, line in enumerate(corpus):
        n = len(line)

        # dimension of the matrix is 2 x n
        score = np.zeros((2, n), dtype=np.float64)
        bptr = np.zeros((2, n), dtype=np.int8)

        ## feature transformation
        X_features = createFeaturesForLine(line, posList[i])
        X_trans = vector.transform(X_features)

        ## generate probabilities for all labels
        observation = clf.predict_log_proba(X_trans)

        ## initialization
        for j in range(2):
            # transition = transitionProb[(100, possibleLabels[j])]
            score[j][0] = observation[0][j]
            bptr[j][0] = 0

        ## iteration
        for t in range(1,n):
            for j in range(2):
                ## calculate max score
                # transition0 = transitionProb[(0, possibleLabels[j])]
                # transition1 = transitionProb[(1, possibleLabels[j])]
                s1 = score[0][t-1]
                s2 = score[1][t-1]
                if s1 > s2:
                    score[j][t] = s1 * observation[t][0]
                    bptr[j][t] = 0
                else:
                    score[j][t] = s2 * observation[t][1]
                    bptr[j][t] = 1

        ## identify sequence
        clf_labels = np.zeros((n), dtype=np.int8)   ## array that stores all labels
        clf_labels[n-1] = 0 if score[0][n-1] > score[1][n-1] else 1
        for j in range(n-2, 0, -1):
            clf_labels[j] = bptr[clf_labels[j+1]][j+1]
        output.append(clf_labels[1:].tolist())
    return output


if __name__ == "__main__":
    ## parse data
    corpus, posList, labelList = getCorpus("./data_release/train.csv")
    validationData = getCorpus("./data_release/val.csv")
    testData = getTestCorpus("./data_release/test_no_label.csv")

    ## create features for training data
    X_train = createFeatures(corpus, posList)
    Y_train = [y for line in labelList for y in line]
    # transition = transitionProbability(labelList, labelsCount(labelList))

    ## use logistic regression model
    model = LogisticRegression()

    ## use dictvectorizer to transform features
    vector = DictVectorizer(sparse=False)
    X_train_trans = vector.fit_transform(X_train)

    ## train model
    clf = model.fit(X_train_trans, Y_train)
    print("labels:", clf.classes_)
    # l = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # for i in l:
    #     print("lambda =", i)
    ## use viterbi to get predictions on validation set
    predictions = viterbi(validationData[0], validationData[1], clf, vector)

    ## compare with actual labels
    true_res = np.array(validationData[2])
    pred_res = np.array(predictions)
    print(labelsCount(true_res.tolist()))
    print(labelsCount(pred_res.tolist()))
    # num_correct = compare(true_res, pred_res)
    # print("%d / %d = %f" %(num_correct, 38628, num_correct/38628))

    outputFile = open('validation-test-m2.csv', 'w')
    outputFile.write('idx,label\n')
    i = 1
    for line in predictions:
        for j in range(len(line)):
            outputFile.write(str(i)+','+str(line[j])+'\n')
            i += 1
    outputFile.close()

    # # output the test result to csv
    # result = viterbi(testData[0], testData[1], clf, vector)
    # outputFile = open('test-result-m2.csv', 'w')
    # outputFile.write('idx,label\n')
    # i = 1
    # for line in result:
    #     for j in range(len(line)):
    #         outputFile.write(str(i)+','+str(line[j])+'\n')
    #         i += 1
    # outputFile.close()
