# encoding: utf-8
"""
FeatureSampler.py is part of ertree an implementation of randomized trees.
Copyright (C) 2009  Jan Hosang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


Created by jan on 2009-06-29.
"""

import sys
import random
from operator import itemgetter
from itertools import takewhile


class HillClimbingFeatureSampler:
    def __init__(self):
        self.__debugMsg = ['hill climber created']
        self.__sample = None

    def setFeatures( self, features ):
        self.__sample = features

    def setNumFeats( self, numFeats ):
        if not self.__sample:
            self.__sample = range( numFeats )
        self.__numFeats = numFeats
        self.__age = {}
        for f in xrange( numFeats ):
            self.__age[f] = 0
        self.__lastScore = None
        self.__scoreDecay = 0.01
        self.__debugMsg = ['hill climber initialised']

    def resample( self, score ):
        self.__debugMsg = []

        if self.__lastScore:
            # check if change was a improvement
            feat = self.__lastChange
            if self.__lastScore < score:
                # if so: keep
                self.__debugMsg.append( 'keeping last change (new score is %f, previous was %f)' % (score,self.__lastScore) )
            else:
                # else:    undo
                self.__debugMsg.append( 'undoing last change (new score is %f, previous was %f)' % (score,self.__lastScore) )
                if feat in self.__sample:
                    self.__sample.remove( feat )
                else:
                    self.__sample.append( feat )
                # find the score of the current featureset
                score = self.__lastScore - self.__scoreDecay
                # because we restored the previous state
            # then reset age of feature
            self.__age[feat] = 0

        # generate a new change (select the oldest feature)
        features = sorted( self.__age.items(), key = itemgetter(1), reverse = True )
        feat, age = features[0]
        features = map( itemgetter(0), takewhile( lambda (f,a): a == age, features ) )
        feat = random.choice( features )
        # delete it if it's in the sample or add it otherwise
        if feat in self.__sample:
            self.__debugMsg.append( 'remove feature %d' % feat )
            self.__sample.remove( feat )
        else:
            self.__debugMsg.append( 'add feature %d' % feat )
            self.__sample.append( feat )
        # remember the changes
        self.__lastChange = feat
        # save the score for the unchanged featureset
        self.__lastScore = score

        # increase age of all features
        for f in self.__age.iterkeys():
            self.__age[f] += 1

        self.__debugMsg.append( 'size of new featureset: %d/%d' % ( len( self.__sample ), self.__numFeats ) )

    def getFeatureSample( self ):
        return self.__sample

    def printStats( self, verbose = False ):
        print >>sys.stderr
        print >>sys.stderr, 'Feature sampler (hill climbing):'
        for msg in self.__debugMsg:
            print >>sys.stderr, '  %s' % msg

class RandomFeatureSampler:
    def __init__( self, n ):
        self.__num = n
        self.__sample = None

    def setFeatures( self, features ):
        if features:
            self.__sample = features
            self.resample( -1.0 )

    def setNumFeats( self, numFeats ):
        if not self.__sample:
            self.__sample = numFeats
            self.resample( -1.0 )

    def resample( self, score ):
        self.__feats = random.sample( self.__sample, self.__num )

    def getFeatureSample( self ):
        return self.__feats

    def printStats( self, verbose = False ):
        pass

class NoFeatureSampler:
    def __init__( self ):
        self.__feats = None

    def setFeatures( self, features ):
        self.__feats = features

    def setNumFeats( self, numFeats ):
        if not self.__feats:
            self.__feats = range( numFeats )

    def resample( self, score ):
        pass

    def getFeatureSample( self ):
        return self.__feats

    def printStats( self, verbose = False ):
        pass


