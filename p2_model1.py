import csv
import numpy as np
import ast

# process file
# format: sentence, pos_seq, label_seq
def getCorpus(filename):
    corpus = []
    labelSeq = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            words = line[0].split()
            labels = ast.literal_eval(line[2])   
            #define the start word <s>
            newWordLine = ['<s>']
            for i in range(len(words)):
                newWordLine.append(words[i])
            corpus.append(newWordLine)

            #define the label start symbol
            newLabelLine = ['<ls>']
            for i in range(len(words)):
                newLabelLine.append(labels[i])
            labelSeq.append(newLabelLine)
    return corpus, labelSeq

# count the number for each label (i.e., number of 0, number of 1)
def labelsCount(labelSeq):
    labels = {}
    for line in labelSeq:
        for i in line:
            if i not in labels:
                labels[i] = 0
            labels[i] += 1
    return labels

# get transition probability between labels (i.e., P(t_i | t_(i-1)))
# each key in this dictionary is in the format (t_(i-1), t_i))
def transitionProbability(labelSeq, labelsCount):
    transitionProb = {}
    for line in labelSeq:
        for i in range(len(line) - 1):
            if (line[i], line[i+1]) not in transitionProb:
                transitionProb[(line[i], line[i+1])] = 0
            transitionProb[(line[i], line[i+1])] += 1
    for w in transitionProb:
        transitionProb[w] = 1.0 * transitionProb[w] / labelsCount[w[0]]
    return transitionProb

# get observation probability (i.e., P(w_i | t_i))
# each key in this dictionary is in the format (t_i, w_i)
def observationProbability(corpus, labelSeq, labelsCount):
    observationProb = {}
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            if (labelSeq[i][j], corpus[i][j]) not in observationProb:
                observationProb[(labelSeq[i][j], corpus[i][j])] = 0
            observationProb[(labelSeq[i][j], corpus[i][j])] += 1
    for w in observationProb:
        observationProb[w] = 1.0 * observationProb[w] / labelsCount[w[0]]
    return observationProb

# bigram viterbi
def viterbi(corpus, transitionProb, observationProb, lambda_=1):
    possibleLabels = [0, 1]
    output = []
    for line in corpus:
        n = len(line)
        # dimension of the matrix is 3 x n
        score = np.zeros((2, n), dtype=np.float64)
        bptr = np.zeros((2, n), dtype=np.int8)

        # compute the score in a log form, (i.e., transition*observation => log(transition)+log(observation))
        # since we just compare them instead of computing the actual score, we don't use exp to recover the probability from log
        # initialization
        for i in range(2):
            transition = transitionProb[('<ls>', possibleLabels[i])]
            try:
                observation = observationProb[(possibleLabels[i], line[1])]
            except KeyError:
                observation = 1e-20
            score[i][1] = lambda_ * np.log(transition) + np.log(observation)
            bptr[i][1] = 0

        # iteration
        for t in range(2, n):
            for i in range(2):
                transition0 = transitionProb[(0, possibleLabels[i])]
                temp0 = score[0][t-1] + lambda_ * np.log(transition0)
                transition1 = transitionProb[(1, possibleLabels[i])]
                temp1 = score[1][t-1] + lambda_ * np.log(transition1)
                maxScore = -1.0
                maxIndex = -1
                if temp0 >= temp1:
                    maxScore = temp0
                    maxIndex = 0
                else:
                    maxScore = temp1
                    maxIndex = 1
                try:
                    observation = observationProb[(possibleLabels[i], line[t])]
                except KeyError:
                    observation = 1e-20
                score[i][t] = maxScore + np.log(observation)
                bptr[i][t] = maxIndex
        
        # identify sequence
        # t is the tag array
        t = np.zeros((n), dtype=np.int8)
        temp0 = score[0][n-1]
        temp1 = score[1][n-1]
        if temp0 >= temp1:
            t[n-1] = 0
        else:
            t[n-1] = 1
        for i in range(n-2, 0, -1):
            t[i] = bptr[t[i+1]][i+1]
        output.append(t[1:].tolist())
    return output
        
def getTestCorpus(filename):
    corpus = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            words = line[0].split()   
            #define the start word <s>
            newWordLine = ['<s>']
            for i in range(len(words)):
                newWordLine.append(words[i])
            corpus.append(newWordLine)
    return corpus

if __name__ == "__main__":
    trainData = getCorpus("./data_release/train.csv")
    labelsCount = labelsCount(trainData[1])
    print(labelsCount)
    transition = transitionProbability(trainData[1], labelsCount)
    #print(transition)
    observation = observationProbability(trainData[0], trainData[1], labelsCount)
    #print(observation)
    
    # output the validation result
    # validationData = getCorpus("./data_release/val.csv")
    # l = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # for i in l:
    #     result = viterbi(validationData[0], transition, observation, i)
    #     outputFile = open('validation-test-%.1f.csv'%i, 'w')
    #     outputFile.write('idx,label\n')
    #     i = 1
    #     for line in result:
    #         for j in range(len(line)):
    #             outputFile.write(str(i)+','+str(line[j])+'\n')
    #             i += 1
    #     outputFile.close()

    # output the test result
    # testData = getTestCorpus("./data_release/test_no_label.csv")
    # result = viterbi(testData, transition, observation, 0.5)
    # outputFile = open('test-result-lambda05.csv', 'w')
    # outputFile.write('idx,label\n')
    # i = 1
    # for line in result:
    #     for j in range(len(line)):
    #         outputFile.write(str(i)+','+str(line[j])+'\n')
    #         i += 1
    # outputFile.close()