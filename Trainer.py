#!/usr/bin/env python
# encoding: utf-8
"""
Trainer.py is part of ertree.

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
import random
import gzip
from itertools import izip, imap, count
from collections import defaultdict

from Boosters import NoBoost
from FeatureSampler import NoFeatureSampler

def myOpen(fn, mode):
    if fn.endswith('.gz'):
        return gzip.open(fn, mode)
    else:
        return _open(fn, mode)
_open = open
open = myOpen

class Voter:
    def __init__( self, classCount, dataCount ):
        self.__votes = [ [0.0] * classCount for i in range( dataCount ) ]
        self.__classCount = classCount
        self.__nonzero        = False

    def add( self, probs, weight = 1.0 ):
        for votes, ps in izip( self.__votes, probs ):
            for i in range( self.__classCount ):
                votes[i] += ps[i] * weight
        self.__nonzero = True

    def getVotes( self ):
        return self.__votes

    def getProbs( self ):
        result = []
        for vs in self.__votes:
            norm = float( sum(vs) )
            result.append( [ v/norm for v in vs ] )
        return result

    def nonzero( self ):
        return self.__nonzero

class Trainer:
    def __init__( self, ModelClass, trainfile, testfiles, probsfile,
                entropyPool = 1, bagging = None, BoosterClass = NoBoost, maxDepth = 1000,
                dumpProbsEach = 10, maxLeafSize = 30, verbose = False,
                featureSampling = NoFeatureSampler(), treeSplit = 'entropy',
                crossvalidation = None, trainprobsfile = None ):
        self.__ModelClass = ModelClass

        self.__trainData  = self.__loadJF( trainfile )
        self.__testData   = [ self.__loadJF( testfile ) for testfile in testfiles ]
        self.__classCount = self.__getClassCount()
        print >>sys.stderr, "%d classes detected" % self.__classCount

        self.__probsfile     = probsfile

        self.__verbose       = verbose
        self.__bagging       = bagging
        self.__maxDepth      = maxDepth
        self.__entropyPool   = entropyPool
        self.__booster       = BoosterClass( self.__trainData, self.__verbose )
        self.__dumpProbsEach = dumpProbsEach
        self.__maxLeafSize   = maxLeafSize
        self.__treeSplit     = treeSplit
        self.__featureSampling = featureSampling
        self.__featureSampling.setNumFeats( len( self.__trainData[0][1] ) )
        self.__crossvalidation = crossvalidation
        self.__trainprobsfile  = trainprobsfile

        self.__trainVotes = Voter( self.__classCount, len( self.__trainData ) )
        self.__testVotes  = [ Voter( self.__classCount, len( testData ) ) for testData in self.__testData ]

    def __buildCV( self ):
        # maybe use a bit more random numbers?
        folds = self.__crossvalidation
        data        = [ self.__trainData[f::folds] for f in xrange( folds ) ]
        origWeights = self.__booster.getWeights()
        weights     = [ origWeights[f::folds] for f in xrange( folds ) ]

        folds = []
        for f in xrange( self.__crossvalidation ):
            trainData = []
            trainWeights = []
            for of in xrange( self.__crossvalidation ):
                if of != f:
                    trainData.extend( data[of] )
                    trainWeights.extend( weights[of] )
            testData = data[f]
            testWeights = weights[f]

            folds.append( ( trainData, trainWeights, testData, testWeights ) )
        return folds

    def __unbuildCV( self, folds ):

        def cycle( folds ):
            iterators = [ (x for x in fold) for fold in folds ]
            n = len( folds )
            for i in count():
                yield iterators[i%n].next()

        n = self.__crossvalidation
        total = sum( imap( len, folds ) )
        #return [ folds[i%n][i//n] for i in xrange( total ) ]
        result = [ x for x in cycle( folds ) ]
        assert len( result ) == total, "WUARGH"
        return result

    def train( self, numModels ):
        i = 1
        try:
            while i <= numModels:
                print >>sys.stderr
                print >>sys.stderr, "Training %s %d..." % ( self.__ModelClass.name(), i )

                if not self.__crossvalidation:
                    trainTestProbs, allTestProbs = self.__trainOnFold(
                                                            self.__trainData, self.__booster.getWeights(),
                                                            self.__trainData, self.__testData, '' )
                    trainProbs = trainTestProbs
                    trainTestData        = self.__trainData
                    trainTestWeights = self.__booster.getWeights()
                    allTestProbs         = [allTestProbs]
                else:
                    # do the n fold cv magic
                    folds = self.__buildCV()
                    trainTestData = []
                    trainTestProbs = []
                    trainTestWeights = []
                    trainProbs = []
                    allTestProbs = []
                    for f, (fTrainTrainData, fTrainTrainWeights, fTrainTestData, fTrainTestWeights) in enumerate( folds ):
                        print >>sys.stderr, '  fold %d/%d' % (f+1, self.__crossvalidation)
                        fTrainTestProbs, fAllTestProbs = self.__trainOnFold(
                                                                fTrainTrainData, fTrainTrainWeights,
                                                                fTrainTestData, self.__testData, '    ' )
                        trainTestData.extend( fTrainTestData )
                        trainTestProbs.extend( fTrainTestProbs )
                        trainTestWeights.extend( fTrainTestWeights )
                        trainProbs.append( fTrainTestProbs )
                        allTestProbs.append( fAllTestProbs )

                    trainProbs = self.__unbuildCV( trainProbs )

                self.__booster.reweight( trainProbs, self.__trainData, trainTestProbs,
                                                                 trainTestData, trainTestWeights, self.__trainVotes )
                for fAllTestProbs in allTestProbs:
                    for testVotes, testProbs in izip( self.__testVotes, fAllTestProbs ):
                        testVotes.add( self.__booster.weightProbs( testProbs ) )

                self.__trainVotes.add( self.__booster.weightProbs( trainProbs ) )

                trainLift = self.__getLift( trainTestProbs, trainTestData )
                self.__featureSampling.resample( trainLift )
                self.__featureSampling.printStats( self.__verbose )

                self.__printStats()

            #
                if self.__dumpProbsEach and ( i % self.__dumpProbsEach ) == 0 and i < numModels:
                    self.dumpProbs( i )

                i += 1
        except KeyboardInterrupt:
            print >>sys.stderr
            print >>sys.stderr, "received interrupt, training stopped"

        featureSample = self.__featureSampling.getFeatureSample()
        print >>sys.stderr
        print >>sys.stderr, "used features:", ' '.join( map( str, sorted( featureSample ) ) )



    def __trainOnFold( self, trainTrainData, trainTrainWeights,
                                         trainTestData, testData, indent = "" ):
        model = self.__ModelClass( self.__classCount, self.__entropyPool,
                                                             self.__maxDepth, self.__maxLeafSize,
                                                             self.__treeSplit )

        if self.__bagging:
            trainTrainData, trainTrainWeights = self.__bag( trainTrainData, trainTrainWeights )

        featureSample = self.__featureSampling.getFeatureSample()

        model.train( trainTrainData, trainTrainWeights, featureSample )

        print >>sys.stderr, "  %s%s" % ( indent, model.stats() )

        print >>sys.stderr, "  %sclassifying data..." % indent
        trainProbs = model.test( trainTestData )
        allTestProbs = [ model.test( testData ) for testData in self.__testData ]

        return trainProbs, allTestProbs

    def __sampleFeatures( self ):
        if not self.__featureSampling:
            return None

        numFeats = len( self.__trainData[0][1] )
        allFeats = xrange( numFeats )
        sampled    = random.sample( allFeats, self.__featureSampling )
        return sampled

    def dumpProbs( self, iteration = None ):
        print >>sys.stderr
        for testVotes, probsfile in izip( self.__testVotes, self.__probsfile ):
            if not testVotes.nonzero():
                continue

            if not iteration:
                filename = probsfile
            else:
                filename = '%s.%d' % ( probsfile, iteration )

            print >>sys.stderr, "dumping probs to %s" % filename

            probs = testVotes.getProbs()
            fp = open( filename, 'w' )
            for ps in probs:
                print >>fp, ' '.join( [ "%.6f" % p for p in ps ] )
            fp.close()

        if self.__trainprobsfile:
            print >>sys.stderr, "dumping trainprobs to %s" % self.__trainprobsfile
            probs = self.__trainVotes.getProbs()
            fp = open( self.__trainprobsfile, 'w' )
            for ps in probs:
                print >>fp, ' '.join( [ "%.6f" % p for p in ps ] )
            fp.close()


    def __bag( self, data, boostWeights ):
        size = self.__bagging

        # with replacement
        indeces = range( len(data) )
        bagIndeces = [ random.choice( indeces ) for i in xrange( int( len(data)*size ) ) ]

        trainData    = [ data[i]         for i in bagIndeces ]
        trainWeights = [ boostWeights[i] for i in bagIndeces ]
        return trainData, trainWeights

    def __printStats( self ):
        print >>sys.stderr, ""
        print >>sys.stderr, "Overall results:"

        trainProbs = self.__trainVotes.getProbs()
        allTestProbs = [ testVotes.getProbs() for testVotes in self.__testVotes ]

        #auc,sepauc = getAUC( trainVotes, trainData, classCount )
        #print >>sys.stderr, "    train AUC:                %f" % auc
        #auc,sepauc = getAUC( votes, testData, classCount )
        #print >>sys.stderr, "    test    AUC:                %f" % auc

        lift = self.__getLift( trainProbs, self.__trainData )
        print >>sys.stderr,     "  train    lift:       %f" % lift
        for i, ( testData, testProbs ) in enumerate( izip( self.__testData, allTestProbs ) ):
            lift = self.__getLift( testProbs,    testData )
            print >>sys.stderr, "  test %3d lift:       %f" % (i+1,lift)

        er = self.__getErrorRate( trainProbs, self.__trainData )
        print >>sys.stderr,     "  train    error rate: %f" % er
        for i, ( testData, testProbs ) in enumerate( izip( self.__testData, allTestProbs ) ):
            er = self.__getErrorRate( testProbs, testData )
            print >>sys.stderr, "  test %3d error rate: %f" % (i+1,er)


    def __getLift( self, allprobs, data ):
        pc1 = len( filter( lambda x: x[0] == 1, data ) )/float(len(data))

        probs = [ (p[1],d[0]) for p,d in izip( allprobs, data ) ]
        probs.sort( key = lambda x: x[0], reverse = True )
        top = int( len( data )*0.2 )
        probs = probs[:top]
        pk1 = len( filter( lambda x: x[1] == 1, probs ) ) / float(top)

        if pc1 == 0:
            return 0.0
        else:
            return pk1/pc1

    def __getAUC( allprobs, data ):

        def getMeasures( c ):
            s = [ sum(c[0]), sum(c[1]) ]
            return float(c[0][0])/s[0], float(c[1][1])/s[1], float(c[0][0]+c[1][1])/(s[0]+s[1])

        classCount = self.__classCount
        auc = []
        cl = [ d[0] for d in data ]

        for c in range( 1, classCount ):
            # target class is set to 1, all others to 0
            classes = map( lambda x: x==c and 1 or 0, cl )

            # read probs from column c
            probs = [ p[c] for p in allprobs ]

            pairs_with_duplicate_probs = zip(probs, classes)
            pairs_with_duplicate_probs.sort()

            pairs = {}

            # accumulate seen zeros while removing duplicates for the initial confusion matrix
            zeros = 0
            for p, c in pairs_with_duplicate_probs:
                    zeros += (1-c)
                    pairs.setdefault( p, [] ).append( c )
            ones = len(pairs_with_duplicate_probs) - zeros

            aucsum = 0
            # initialize confusion matrix
            c=[[0,0],[0,0]]
            c[0][1] = zeros
            c[1][1] = ones

            # initialize first point on the auc plane
            sens, spec, prec = getMeasures(c)
            spec_prev, sens_prev = spec, sens

            # for each threshold teh confusion matrix may change
            # depending only on the ammount of zeros and ones assigned to the probability
            for k in sorted(pairs.keys()):
                    v = pairs[k]
                    ones = sum(v)
                    zeros = len(v) - ones
                    c[0][0] += zeros
                    c[0][1] -= zeros
                    c[1][0] += ones
                    c[1][1] -= ones
                    # calculate new point for the auc plane
                    sens, spec, prec = getMeasures(c)

                    # add area to auc. sens = x-axis, spec = y-axis
                    width = (sens - sens_prev)
                    aucsum += spec * width
                    aucsum += width * 0.5 * (spec_prev - spec)

                    # memorize the last point for the next area-calculation
                    spec_prev, sens_prev = spec, sens

            auc.append( aucsum )

        return sum(auc)/len(auc), auc

    def __getErrorRate( self, probs, data ):
        errors = 0
        for ps, d in izip( probs, data ):
            k = d[0]
            maxC = -1
            maxP = -1
            for c, p in enumerate( ps ):
                if p > maxP:
                    maxP = p
                    maxC = c
            if maxC != k:
                errors += 1
        return float(errors) / len(data)

    def __loadJF( self, filename ):
        print >>sys.stderr, "reading %s..." % filename
        result = []
        f = open( filename, "r" )
        f.readline()
        for line in f:
            if line.startswith( '-1' ):
                break
            tmp = line.split()
            result.append( ( int(tmp[0]), map( float, tmp[1:] ) ) )
        f.close()
        print >>sys.stderr, "  read %d lines" % len( result )

        return result

    def __getClassCount( self ):
        classes = {}
        for c, feats in self.__trainData:
            classes[c] = 1
        for testData in self.__testData:
            for c, feats in testData:
                classes[c] = 1
        return len( classes )




