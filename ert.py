#!/usr/bin/env python
# encoding: utf-8
"""
ert.py is part of ertree.

ertree is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License 3 as published by
the Free Software Foundation.

ertree is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ertree.  If not, see <http://www.gnu.org/licenses/>.


Created by jan on 2009-06-04.

"""

import sys
import getopt

from Trainer import Trainer
from Boosters import *
from Models import *
from FeatureSampler import *

help_message = '''
The help message goes here.

    -h|--help   display this usage
    -v          be verbose
    -t|--train  the data in joergfile format to train models on
    -T|--test   the data in joergfile format to extract probabilities on;
                you can pass a comma seperated list, if you want to evaluate
                the models on multiple test sets, but remember to pass a list
                for probs output as well
    --probs     file to dump probabilities of the testset to (plain columns)
    -e|--ent|--entropy
                specify how many random tests to generate in each tree node
                (the one with the best information gain will be chosen)
    --leafsize  stop seperating data in a node if its total weight is less
                than leafsize/N; in case of no boosting: stop separation
                if there are less than leafsize examples in a node
    -d|--maxdepth
                maximum depth of trees; default: 1000
    -s|--treesplit
                how to determine split of an attribute:
                entropy (default), random
    -c|--count|--iter
                how many models are to be trained
    --model     the type of model to train: tree, forest
    --forestsize
                if the model is forest, this argument specifies how many
                trees to train in it
    --boost     the booster to use: none (default), ada, lift, auc
    --bagging   train only on some random part of the training set (eg. 0.8);
                in case of boosting the remaining data will be used to
                estimate the weighted error of a model
    --fs|--featuresampling
                only consider a certain amount of randomly determined
                features; specify the amount of features to consider;
                set to "hill" for hill-climbing feature selection
'''


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    global usingPsyco

    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ht:T:e:c:d:s:v",
                    ["help", "train=", "test=", "entropy=", "ent=", "probs=",
                     "boost=", "count=", "iter=", "model=", "forestsize=",
                     "bagging=", "leafsize=", "maxdepth=", "treesplit=",
                     "fs=", "featuresampling=", "cv=", "crossvalidation=",
                     "features=", "trainprobs="])
        except getopt.error, msg:
            raise Usage(msg)

        verbose     = False
        trainfile = None
        testfile    = None
        probsfile = None
        entropy     = 1
        booster     = NoBoost
        maxDepth    = 1000
        bagging     = None
        leafsize    = 30.0
        features    = None
        trainprobsfile    = None
        dumpProbsEach     = None
        modelCount            = 1
        modelClass            = Tree
        treesplit             = 'entropy'
        featuresampling = NoFeatureSampler()
        crossvalidation = None
        # option processing
        for option, value in opts:
            if option == "-v":
                verbose = True
            if option in ("-h", "--help"):
                raise Usage(help_message)
            if option in ("-t", "--train"):
                trainfile = value
            if option in ("-T", "--test"):
                testfile = value.split( ',' )
            if option in ("--probs"):
                probsfile = value.split( ',' )
            if option in ("-e", "--entropy", "--ent"):
                entropy = int( value )
            if option in ("-c", "--count", "--iter"):
                modelCount = int( value )
            if option in ("--model"):
                if value in ("tree", "ert"):
                    modelClass = Tree
                if value in ("forest"):
                    modelClass = Forest
            if option in ("--forestsize"):
                Forest.size = int( value )
            if option in ("--boost"):
                if value == "none":
                    booster = NoBoost
                elif value == "ada":
                    booster = AdaBoost
                elif value == "lift":
                    booster = LiftBoost
                elif value in ("softrank", "auc"):
                    booster = SoftRankBoost
                else:
                    raise Usage( "unknown booster '%s'" % value )
            if option in ("--bagging"):
                bagging = float( value )
            if option in ("--leafsize"):
                leafsize = float( value )
            if option in ("-d", "--maxdepth"):
                maxDepth = int( value )
            if option in ("-s", "--treesplit"):
                if value in ("ent", "entropy"):
                    treesplit = "entropy"
                elif value in ("rand", "random"):
                    treesplit = "random"
                else:
                    raise Usage( "unknown treesplit method '%d'" % value )
            if option in ("--fs", "--featuresampling"):
                if value == 'hill':
                    featuresampling = HillClimbingFeatureSampler()
                else:
                    featuresampling = RandomFeatureSampler( int( value ) )
            if option in ("--cv", "--crossvalidation"):
                crossvalidation = int( value )
            if option in ("--features"):
                features = map( int, value.split(',') )
            if option in ("--trainprobs"):
                trainprobsfile = value

        if not trainfile:
            raise Usage( "no trainfile specified" )
        if not testfile:
            raise Usage( "no testfile specified" )
        if not probsfile:
            raise Usage( "no probsfile specified" )

        if len( testfile ) != len( probsfile ):
            raise Usage( "passed %d test sets, but %d probsfiles" % ( len( testfile ), len( probsfile ) ) )

        if verbose:
            if usingPsyco:
                print >>sys.stderr, "using full psyco"
                print >>sys.stderr
            else:
                print >>sys.stderr, "psyco not found"
                print >>sys.stderr

    except Usage, err:
        print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        print >> sys.stderr, "    for help use --help"
        return 2

    featuresampling.setFeatures( features )
    trainer = Trainer( modelClass, trainfile, testfile, probsfile,
                       entropy, bagging, booster, maxDepth, dumpProbsEach,
                       leafsize, verbose, featuresampling, treesplit,
                       crossvalidation, trainprobsfile )
    trainer.train( modelCount )
    trainer.dumpProbs()

if __name__ == "__main__":
    try:
        import psyco
        psyco.full()
        usingPsyco = True
    except ImportError:
        usingPsyco = False

    sys.exit(main())
