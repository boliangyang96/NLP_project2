# CS 4740 Fall 2019 Project 2 -- metaphor detection with sequence labeling

Please read the PDF writeup for P2 for detailed requirements

## Data
In the folder data_release, we include the train/dev/test.csv files -- train.csv and dev.csv come with labels, and test.csv doesn't

## Eval

```
python eval.py --pred sample_out.csv

(replace sample_out.csv with 'your prediction file name')
```

Submit your prediction file for test.csv to kaggle, it should have the same format with sample_out.csv

Hint: you can refer to eval.py for how to write the code for reading the input files.
