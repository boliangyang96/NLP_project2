For model1:

functions (details of the function is written in comments): getCorpus, labelsCount, wordsCount, transitionProbability, observationProbability, observationProbabilityWithAddK, viterbi, getTestCorpus;
1. use getCorpus, labelsCount, transitionProbability, and observationProbability to get the courpus, count of the labels, trainsition probability and the observation probability of the training dataset;
2. use getCorpus/getTestCorpus to get the validation/test dataset and apply viterbi to get the result
How to run: there are three blocks in the main function, each of them is commented out.
If want to run, simply uncomment any one of the blocks, and just run 'python p2_model1.py'. The first block is lambda experiment on validaiton dataset; the second is add-k smoothing experiment on validation dataset; the third is to output the test result

======================================================================================================================
For model2:

Functions
1. Use getCorpus() to process train and validation data to get corpus(list), pos(list), labels(list)
   Use getTestCorpus() to process test data without labels
2. Use createFeatures() to get the features for the corpus
3. Use observationProbability() defined in model1 to get the observation probability
4. Create logisticRegression model to fit the training data
5. Use viterbi() to get predictions on validation/test data

How to run
- dependency: sklearn library, numpy library, p2_model1.py
- run: python3 p2_model2.py
- add features: in createFeatures() and createFeaturesForLine() functions, there are commented lines such as "pos_prev_feature.append(prev_pos2)" and "feature["prev_pos2"] = prev_pos2[i][j]", each pair of these lines represent a feature. When uncommenting lines related to prev_pos2 in createFeatures(), similar lines in createFeaturesForLine() related to prev_pos2 should also be uncommented.
- remove features: comment out the lines
- feature explanation
  - prev_pos: the pos tag of the first word preceding the current one
  - prev_pos2: the pos tag of the second word preceding the current one
  - next_pos, next_pos2: similar to above, the pos tag of words following the current one
  - prev_word: the first word preceding the current one
  - next_word: the first word following the current one
- Note: each feature takes a large amount of memory, add too many features or running on computers without enough memory may result in segmentation fault error.