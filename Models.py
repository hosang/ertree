#!/usr/bin/env python
# encoding: utf-8
"""
Models.py is part of ertree an implementation of randomized trees.
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


Created by jan on 2009-06-04.
"""

import sys
import random
import math
from collections import defaultdict
from itertools import izip, ifilter, imap
from operator import itemgetter
from bisect import bisect_left

class Forest:
    def __init__( self, classCount, entropyPool, maxDepth, maxLeafSize, treeSplit ):
        self.__entropyPool = entropyPool
        self.__classCount = classCount
        self.__maxDepth = maxDepth
        self.__maxLeafSize = maxLeafSize
        self.__size = Forest.size
        self.__treeSplit = treeSplit

    @staticmethod
    def name():
        return "extremely randomized forest"

    def train( self, data, weights, validFeats ):
        self.__trees = []
        print >>sys.stderr, "    tree",
        for i in xrange( self.__size ):
            print >>sys.stderr, (i+1),
            t = Tree( self.__classCount, self.__entropyPool, self.__maxDepth, self.__maxLeafSize, self.__treeSplit )
            t.train( data, weights, validFeats )
            self.__trees.append( t )
        print >>sys.stderr

    def test( self, data ):
        votes = [ [0.0]*self.__classCount for i in xrange( len( data ) ) ]
        for t in self.__trees:
            cs = t.test( data )
            for c,vote in izip( cs, votes ):
                for k in xrange( len( c ) ):
                    vote[k] += c[k]

        for vs in votes:
            norm = float(sum(vs))
            for i in xrange( len( vs ) ):
                vs[i] /= norm

        return votes

    def stats( self ):
        maxd = -1
        mind = 100000
        avgd = 0
        for t in self.__trees:
            if t.depth() > maxd: maxd = t.depth()
            if t.depth() < mind: mind = t.depth()
            avgd += t.depth()
        avgd /= float( len( self.__trees ) )
        return "depth (min-max/avrg): %d-%d/%d" % ( mind, maxd, avgd )

class Tree:
    def __init__(self, classCount = 2, entropyPool = 1, maxDepth = 1000, maxLeafSize = 30,
                             treeSplit = 'entropy' ):
        self.__maxLeafSize = maxLeafSize
        self.__entropyPool = entropyPool
        self.__classCount    = classCount
        self.__maxDepth        = maxDepth
        self.__depth             = 0
        self.__treeSplit     = treeSplit

    @staticmethod
    def name():
        return "extremely randomized tree"

    def stats( self ):
        return "depth: %d" % self.__depth

    def depth( self ):
        return self.__depth

    def test( self, data ):
        assert self.__numFeats == len( data[0][1] ), "number features (%d) differ from train (%d)" % ( self.__numFeats, len( data[0][1] ) )
        return [ self.__test( feats ) for c,feats in data ]

    def __test( self, feats ):
        tree = self.__tree
        #while isinstance( tree, tuple ):
        while type( tree ) == tuple:
            ( lefttree, test, righttree ) = tree
            if feats[ test[0] ] <= test[1]:
                tree = lefttree
            else:
                tree = righttree
        return tree

    def train( self, data, boostWeights, validFeats = None ):
        self.__maxWeightedLeafSize = float( self.__maxLeafSize ) / len(data)
        self.__numFeats = len( data[0][1] )
        self.__numData    = len( data )
        if not validFeats:
            validFeats = range( self.__numFeats )
        self.__tree = self.__train( data, boostWeights, self.__getContainedClasses( data ), 0, validFeats )

    def __getContainedClasses( self, data ):
        classes = {}
        for c, f in data:
            classes[c] = 1
        return classes.keys()

    def __train( self, data, weights, containedClasses, depth, validFeats ):
        if depth > self.__depth:
            self.__depth = depth
        if depth >= self.__maxDepth or sum( weights ) <= self.__maxWeightedLeafSize:
        #if depth > self.__maxDepth or len( data ) <= self.__maxLeafSize:
            return self.__getMajorClass( data, weights )

        if len( containedClasses ) == 1:
            result = [0] * self.__classCount
            result[ containedClasses[0] ] = 1
            return result

        validFeats = validFeats[:]
        maxInfoGain = -1000000
        maxTest = None
        tests = self.__generateTests( data, weights, self.__entropyPool, validFeats )
        for test, infogain in tests:
            #( leftdata, leftweights, rightdata, rightweights ) = self.__applyTest( test, data, weights )
            #infogain = self.__informationGain( leftdata, leftweights, rightdata, rightweights )
            if infogain > maxInfoGain:
                maxInfoGain = infogain
                maxTest = test
                #maxData = ( leftdata, leftweights, rightdata, rightweights )

        if maxTest == None:
            return self.__getMajorClass( data, weights )

        #( leftdata, leftweights, rightdata, rightweights ) = maxData
        ( leftdata, leftweights, leftclasses, rightdata, rightweights, rightclasses ) = self.__applyTest( maxTest, data, weights )
        return ( self.__train( leftdata, leftweights, leftclasses, depth+1, validFeats ), maxTest,
                         self.__train( rightdata, rightweights, rightclasses, depth+1, validFeats ) )

    def __getMajorClass( self, data, weights ):
        #global cumClassScore

        classes = [0.0] * self.__classCount

        for d, w in zip( data, weights ):
            c, feats = d
            classes[c] += w
            #classes[c] = classes.setdefault(c,0) + w

        norm = float( sum( classes ) )
        return [ c/norm for c in classes ]
        #maxScore = -1000000
        #maxClass = -1
        #for c, cnt in classes.items():
        #    score = cumClassScore[c] * cnt
        #    #score = cnt
        #    if score > maxScore:
        #        maxScore = score
        #        maxClass = c
        #
        #return maxClass

    def __getThreshold( self, data, weights, f, needIG, rand ):

        def cumulate( cumArray, addArray ):
            #return map( sum, izip( cumArray, addArray ) )
            for i, inc in enumerate( addArray ):
                cumArray[i] += inc

        def entropy( classCnt, N ):
                result = 0.0
                for cnt in classCnt:
                    if cnt > 0:
                        p = cnt/N
                        result -= p * math.log( p, 2 )
                return result

        def informationGain( leftClassCnt, rightClassCnt ):
            leftCnt    = float( sum( leftClassCnt ) )
            rightCnt = float( sum( rightClassCnt ) )
            N = leftCnt + rightCnt
            return - leftCnt/N * entropy( leftClassCnt, leftCnt ) \
                         - rightCnt/N * entropy( rightClassCnt, rightCnt )

        def countValues( data, weights, f ):

            def repeat( obj ):
                while True:
                    yield obj[:]

            values = defaultdict( repeat( [0.0]*self.__classCount ).next )
            for (c,feats), w in izip( data, weights ):
                values[feats[f]][c] += w
            return values

        values = countValues( data, weights, f )

        if len( values ) <= 1:
            return None

        values = values.items()
        values.sort()

        for i in xrange( 1,    len( values ) ):
            cumulate( values[i][1], values[i-1][1] )

        maxig = -1000000
        maxthr = -1

        if rand:
            eps = 0.0001
            a = values[0][0]
            b = values[-1][0] - eps

            thr = random.uniform( a, b )
            i = bisect_left( map( itemgetter(0), values ), maxthr )
            right = [ values[-1][1][c] - values[i][1][c] for c in xrange( self.__classCount ) ]

            maxig    = informationGain( values[i][1], right )
            maxthr = i

        else:
            for i in xrange( len( values )-1 ):
                right = [ values[-1][1][c] - values[i][1][c] for c in xrange( self.__classCount ) ]
                ig = informationGain( values[i][1], right )
                if ig > maxig:
                    maxig = ig
                    maxthr = i

        return values[ maxthr ][0], maxig

    def __generateTests( self, data, weights, n, feats ):
        randSplit = self.__treeSplit == 'random'

        needIG = n > 1
        result = []
        myfeats = feats[:]
        while len( result ) < n and len( myfeats ) > 0:
            i = random.randint( 0, len( myfeats ) - 1 )
            feat = myfeats[i]
            if self.__treeSplit == 'entropy':
                # split is optimal, no need to generate another test for
                # that feature again in this node
                del myfeats[i]

            test = self.__getThreshold( data, weights, feat, needIG, randSplit )
            if test != None:
                threshold, ig = test
                result.append( ( ( feat, threshold ), ig ) )
            else:
                #del feats[i]
                feats.remove( feat )
                if self.__treeSplit == 'random':
                    del myfeats[i]
        return result

    def __applyTest( self, test, data, weights ):
        left   = [ d for d in ifilter( lambda( c, f ): f[test[0]] <= test[1], data ) ]
        right  = [ d for d in ifilter( lambda( c, f ): f[test[0]] >  test[1], data ) ]

        leftW  = [ w for d, w in ifilter( lambda( (c,f), w ): f[test[0]] <= test[1], izip( data, weights ) ) ]
        rightW = [ w for d, w in ifilter( lambda( (c,f), w ): f[test[0]] >  test[1], izip( data, weights ) ) ]

        leftClasses  = list( set( [ c for c, f in left  ] ) )
        rightClasses = list( set( [ c for c, f in right ] ) )

        return ( left, leftW, leftClasses, right, rightW, rightClasses )

    def __informationGain( self, leftdata, leftweights, rightdata, rightweights ):
        s1 = sum( leftweights )
        s2 = sum( rightweights )
        N = float( s1+s2 )
        p1 = s1/N
        p2 = s2/N
        return - p1 * self.__entropy( leftdata, leftweights ) - p2 * self.__entropy( rightdata, rightweights )

    def __entropy( self, data, weights ):
        classes = [0]*self.__classCount
        N = 0.0
        for d, w in zip( data, weights ):
            c, feats = d
            classes[c] += w
            N += w
        sum = 0.0
        for c in xrange( self.__classCount ):
            if classes[c] > 0:
                p = classes[c]/N
                sum -= p * math.log( p, 2 )
        return sum


