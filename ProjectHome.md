# Randomized Decision Trees #

Ertree are more or less extremely randomized decision trees implemented in python. It lets you choose how randomized the training will be and which methods to use.

## Flavor ##

```
$ ../ert.py -t train.jf.gz -T test.jf.gz -e 5 -s ent --leafsize 5 -c 50 --probs test.probs
reading train.jf.gz...
  read 10000 lines
reading test.jf.gz...
  read 17000 lines
2 classes detected

[...]

Training extremely randomized tree 50...
  depth: 41
  classifying data...

Overall results:
  train    lift:       5.000000
  test   1 lift:       3.962766
  train    error rate: 0.000600
  test   1 error rate: 0.020941
```

This is an example run on a small subset of UCSD 2009 data. The data is included in the svn repository.


## Features ##

Ertree lets you choose among a lot of features, so thou shallst win ye contest. Missing something? This is open source, this is python, add it! (see below)

Ertree trains some amount of models, between the training of each model it does boosting, bagging and feature sampling/selection. A model might be a tree, but if you are want to use extremely randomized trees, it is a bad idea to boost after every tree, because one tree for itself is really, really bad, so you can train a set of tree, a forest, instead.

#### Tree Growing Options ####
  * **Information gain tests**: Choose how may tests are to be performed per node before the optimal test is being chosen.
  * **Leaf size**: Choose the maximum number of data items per node before the splitting is being stopped.
  * **Maximum depth**: A tree never grows taller than this.
  * **Test generation**: Split points for tests per feature can be chosen randomly or optimally according to the information gain.

#### Model Types ####
  * **Quantity**: How many models will be trained. This determines how many training/boosting iterations will be done.
  * **Type**: You can either train a single tree or a set of trees.

#### Boosting ####
I implemented some boosting algorithms which are probably crude and broken, especially if it comes to multi class problems: Ada, Lift and Area Under Curve (AUC).

#### Bagging ####
You can enable bagging and specify how much data will be sampled for training. The remaining training data will be used to estimate the training error for boosting.

#### Feature Sampling ####
It's possible to let the features be determined randomly or by hill climbing.



## Proof of Performance ##

This piece of code was used to create large parts of the winning submissions for the students track of the [UCSD Data Mining Contest 2009](http://mill.ucsd.edu/index.php?page=History).

![http://www.janhosang.com/images/thisisdata.jpg](http://www.janhosang.com/images/thisisdata.jpg)


## Get Involved! ##

You want to do a bit of data mining for fun? **Try it!** Certainly, you will notice the program does not do what you want, is written awfully and sucks in general. **Tell me** about your experience or **send me patches**! Drop me a mail: jan [dot](dot.md) hosang [at](at.md) gmail [dot](dot.md) com.

The program changed a lot during the contest as I tried to improve speed and score, so the code might be a big mess. I will try to change that so the project gets friendlier.

Have a look at the ToDo wiki page.