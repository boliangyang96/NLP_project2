import csv
import numpy as np
import ast
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from p2_model1 import labelsCount, observationProbability

""" Extract data from file
    return: corpusList, posList, labelList
"""
def getCorpus(filename):
    corpus = []
    posSeq = []
    labelSeq = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            words = line[0].split()
            newWordLine = []
            for i in range(len(words)):
                newWordLine.append(words[i])
            corpus.append(newWordLine)

            pos = ast.literal_eval(line[1])
            newPosLine = []
            for i in range(len(pos)):
                newPosLine.append(pos[i])
            posSeq.append(newPosLine)

            # define the label start symbol
            labels = ast.literal_eval(line[2])
            newLabelLine = []
            for i in range(len(labels)):
                newLabelLine.append(labels[i])
            labelSeq.append(newLabelLine)
    return corpus, posSeq, labelSeq

""" Extract test data without label
    return: corpusList, posList
"""
def getTestCorpus(filename):
    corpus = []
    posSeq = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            words = line[0].split()
            newWordLine = []
            for i in range(len(words)):
                newWordLine.append(words[i])
            corpus.append(newWordLine)

            pos = ast.literal_eval(line[1])
            newPosLine = []
            for i in range(len(pos)):
                newPosLine.append(pos[i])
            posSeq.append(newPosLine)
    return corpus, posSeq

""" features created
        1. current word
        2. current pos tag
        3. previous pos tag (window=3)
        4. following pos tag (window=3)
        5. previous word (window=1)
        6. following word (window=1)
"""
def createFeatures(corpus, posList):
    ## feature: previous word's pos tag
    pos_prev_feature = list()
    prev_pos = copy.deepcopy(posList)
    prev_pos2 = copy.deepcopy(posList)
    prev_pos3 = copy.deepcopy(posList)
    pos_prev_feature.append(prev_pos)
    # pos_prev_feature.append(prev_pos2)
    # pos_prev_feature.append(prev_pos3)
    ## feature: next word's pos tag
    pos_next_feature = list()
    next_pos = copy.deepcopy(posList)
    next_pos2 = copy.deepcopy(posList)
    next_pos3 = copy.deepcopy(posList)
    pos_next_feature.append(next_pos)
    # pos_next_feature.append(next_pos2)
    # pos_next_feature.append(next_pos3)
    ## feature: previous word
    word_prev_feature = list()
    prev_word = copy.deepcopy(corpus)
    prev_word2 = copy.deepcopy(corpus)
    prev_word2 = copy.deepcopy(corpus)
    word_prev_feature.append(prev_word)
    # word_prev_feature.append(prev_word2)
    # word_prev_feature.append(prev_word3)
    ## feature: next word
    word_next_feature = list()
    next_word = copy.deepcopy(corpus)
    next_word2 = copy.deepcopy(corpus)
    next_word3 = copy.deepcopy(corpus)
    word_next_feature.append(next_word)
    # word_next_feature.append(next_word2)
    # word_next_feature.append(next_word3)
    ## merge features into list of dicts
    X_features = list()
    for i, sentence in enumerate(corpus):
        for n, f_list in enumerate(pos_prev_feature):
            for m in range(n+1):
                f_list[i].insert(0, "None")
                f_list[i].pop()

        for n, f_list in enumerate(pos_next_feature):
            for m in range(n+1):
                f_list[i].append("None")
                f_list[i].pop(0)

        for n, f_list in enumerate(word_prev_feature):
            for m in range(n+1):
                f_list[i].insert(0, "Start")
                f_list[i].pop()

        for n, f_list in enumerate(word_next_feature):
            for m in range(n+1):
                f_list[i].append("End")
                f_list[i].pop(0)


        for j, word in enumerate(sentence):
            feature = dict()
            feature["word"] = word
            feature["pos"] = posList[i][j]
            feature["prev_pos"] = prev_pos[i][j]
            feature["next_pos"] = next_pos[i][j]
            # feature["prev_pos2"] = prev_pos2[i][j]
            # feature["next_pos2"] = next_pos2[i][j]
            # feature["prev_pos3"] = prev_pos3[i][j]
            # feature["next_pos3"] = next_pos3[i][j]
            feature["prev_word"] = prev_word[i][j]
            feature["next_word"] = next_word[i][j]
            X_features.append(feature)
    return X_features

""" Similar to createFeatures, this one only process a line
"""
def createFeaturesForLine(line, posList):
    ## feature: previous word's pos tag
    pos_prev_feature = list()
    prev_pos = copy.deepcopy(posList)
    prev_pos2 = copy.deepcopy(posList)
    prev_pos3 = copy.deepcopy(posList)
    pos_prev_feature.append(prev_pos)
    # pos_prev_feature.append(prev_pos2)
    # pos_prev_feature.append(prev_pos3)
    for n, f_list in enumerate(pos_prev_feature):
        for m in range(n+1):
            f_list.insert(0, "None")
            f_list.pop()
    # feature: next word's pos tag
    pos_next_feature = list()
    next_pos = copy.deepcopy(posList)
    next_pos2 = copy.deepcopy(posList)
    next_pos3 = copy.deepcopy(posList)
    pos_next_feature.append(next_pos)
    # pos_next_feature.append(next_pos2)
    # pos_next_feature.append(next_pos3)
    for n, f_list in enumerate(pos_next_feature):
        for m in range(n+1):
            f_list.append("None")
            f_list.pop(0)
    # feature: previous word
    word_prev_feature = list()
    prev_word = copy.deepcopy(line)
    prev_word2 = copy.deepcopy(line)
    prev_word2 = copy.deepcopy(line)
    word_prev_feature.append(prev_word)
    for n, f_list in enumerate(word_prev_feature):
        for m in range(n+1):
            f_list.insert(0, "Start")
            f_list.pop()
    # feature: next word
    word_next_feature = list()
    next_word = copy.deepcopy(line)
    next_word2 = copy.deepcopy(line)
    next_word3 = copy.deepcopy(line)
    word_next_feature.append(next_word)
    for n, f_list in enumerate(word_next_feature):
        for m in range(n+1):
            f_list.append("End")
            f_list.pop(0)

    ## merge features into list
    X_features = list()
    for i, word in enumerate(line):
        feature = dict()
        feature["word"] = word
        feature["pos"] = posList[i]
        feature["prev_pos"] = prev_pos[i]
        feature["next_pos"] = next_pos[i]
        # feature["prev_pos2"] = prev_pos2[i]
        # feature["next_pos2"] = next_pos2[i]
        # feature["prev_pos3"] = prev_pos3[i]
        # feature["next_pos3"] = next_pos3[i]
        feature["prev_word"] = prev_word[i]
        feature["next_word"] = next_word[i]
        X_features.append(feature)
    return X_features

# bigram viterbi
def viterbi(corpus, posList, clf, vector, observationProb):
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
        label_prob = clf.predict_log_proba(X_trans)

        ## initialization
        for j in range(2):
            try:
                observation = observationProb[(possibleLabels[j], line[0])]
            except KeyError:
                observation = 1e-20
            observation = np.log(observation)

            score[j][0] = label_prob[0][j] + observation
            bptr[j][0] = 0

        ## iteration
        for t in range(1,n):
            for j in range(2):
                ## calculate max score
                s1 = score[0][t-1] + label_prob[t][j]
                s2 = score[1][t-1] + label_prob[t][j]
                try:
                    observation = observationProb[(possibleLabels[j], line[t])]
                except KeyError:
                    observation = 1e-20
                observation = np.log(observation)

                if s1 >= s2:
                    score[j][t] = s1 + observation
                    bptr[j][t] = 0
                else:
                    score[j][t] = s2 + observation
                    bptr[j][t] = 1

        ## identify sequence
        clf_labels = np.zeros((n), dtype=np.int8)   ## array that stores all labels
        clf_labels[n-1] = 0 if score[0][n-1] > score[1][n-1] else 1
        for j in range(n-2, 0, -1):
            clf_labels[j] = bptr[clf_labels[j+1]][j+1]
        output.append(clf_labels.tolist())
    return output


if __name__ == "__main__":
    ## parse data
    corpus, posList, labelList = getCorpus("./data_release/train.csv")
    validationData = getCorpus("./data_release/val.csv")
    testData = getTestCorpus("./data_release/test_no_label.csv")

    ## create features for training data
    X_train = createFeatures(corpus, posList)
    Y_train = [y for line in labelList for y in line]

    ## observation probabilty from model 1
    observation = observationProbability(corpus, labelList, labelsCount(labelList))

    ## use logistic regression model
    model = LogisticRegression(multi_class='auto')

    ## use dictvectorizer to transform features
    vector = DictVectorizer(sparse=False)
    X_train_trans = vector.fit_transform(X_train)

    ## train model
    clf = model.fit(X_train_trans, Y_train)

    ## use viterbi to get predictions on validation set
    predictions = viterbi(validationData[0], validationData[1], clf, vector, observation)

    ## compare with actual labels
    true_res = np.array(validationData[2])
    pred_res = np.array(predictions)

    print(labelsCount(true_res.tolist()))
    print(labelsCount(pred_res.tolist()))

    ## output validation result to csv
    outputFile = open('validation-test-m2.csv', 'w')
    outputFile.write('idx,label\n')
    i = 1
    for line in predictions:
        for j in range(len(line)):
            outputFile.write(str(i)+','+str(line[j])+'\n')
            i += 1
    outputFile.close()

    # output test result to csv
    result = viterbi(testData[0], testData[1], clf, vector, observation)
    outputFile = open('test-result-m2.csv', 'w')
    outputFile.write('idx,label\n')
    i = 1
    for line in result:
        for j in range(len(line)):
            outputFile.write(str(i)+','+str(line[j])+'\n')
            i += 1
    outputFile.close()
