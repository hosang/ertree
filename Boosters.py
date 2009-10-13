#!/usr/bin/env python
# encoding: utf-8
"""
Boosters.py is part of ertree.

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
import math
from itertools import izip, imap, ifilter, dropwhile
from operator import itemgetter

class SoftRankBoost:
    def __init__( self, data, verbose = False ):
        p = float( len( filter( lambda d: d[0] == 1, data ) ) )
        n = len( data ) - p

        #p = 1/p
        #n = 1/n
        p = 0.5/p
        n = 0.5/n
        self.__weights = [ p if c == 1 else n for c,f in data ]

        self.__gammaHat = -2        # minus infinity
        self.__delta        = 0.5     # SoftRankBoost.delta
        self.__alphaSum = 0.0

    def getWeights( self ):
        return self.__weights

    def __getGamma( self, probs, data, weights ):
        # gamma = 1     means good classification
        # gamma = -1    means bad    classification
        #return sum( w*(p[c]*2-1) for p, (c,f), w in izip( probs, data, weights ) ) / 2
        return sum( w*(p[c]*2-1) for p, (c,f), w in izip( probs, data, weights ) )

    def __getBias( self, probs, data, weights ):
        # positive classifications get positive sign (inside of the sum)
        # negative classifications get a negative sign
        #return - sum( w*(p[1]*2-1) for p, w in izip( probs, weights ) ) / 2
        return - sum( w*(p[1]*2-1) for p, w in izip( probs, weights ) )

    def reweight( self, probs, data, modelTrainProbs, modelTrainData, modelTrainWeights, voter ):
        print >>sys.stderr, "    boosting (SoftRankBoost)"

        gamma = self.__getGamma( modelTrainProbs, modelTrainData, modelTrainWeights )
        self.__gammaHat = min( self.__gammaHat, (1-self.__delta)*gamma )
        print >>sys.stderr, "        weighted error: %f" % (1-(gamma+1)/2.0)

        self.__alpha = ( gamma - self.__gammaHat ) / ( 2 * ( 1 + self.__gammaHat ** 2 ) )
        self.__alphaSum += self.__alpha
        print >>sys.stderr, "        alpha:                    %f" % self.__alpha


        self.__b = self.__getBias( modelTrainProbs, modelTrainData, modelTrainWeights )

        # now reweight
        gammaTerm = self.__gammaHat * self.__alphaSum
        currentVotes = self.weightProbs( probs )
        norm = [ 0.0, 0.0 ]
        for i, (votes, cVotes, (c, f)) in enumerate( izip( voter.getVotes(), currentVotes, data ) ):
            # negative if classification right
            # positive if classification wrong
            # not sure if it is the right translation from (p0,p1) to the paper's
            # [-1,1] notation of f
            x = (votes[1-c]+cVotes[1-c])-(votes[c]+cVotes[c]) + gammaTerm
            if x < 0:
                self.__weights[i] = math.exp(x)
            else:
                self.__weights[i] = 1.0

            norm[c] += self.__weights[i]

        # normalise
        norm = [ n*2.0 for n in norm ]
        self.__weights = [ w/norm[c] for w,(c,f) in izip( self.__weights, data ) ]

    def weightProbs( self, probs ):
        offset = self.__b/2.0
        return [ [ self.__alpha*(p[0]-offset)/2, self.__alpha*(p[1]+offset)/2 ] for p in probs ]

class LiftBoost:
    def __init__( self, data, verbose = False ):
        dataCount = len( data )
        self.__weights = [ 1.0/dataCount ] * dataCount
        self.__alpha     = 1.0
        self.__verbose = verbose

    def getWeights( self ):
        return self.__weights

    def weightProbs( self, probs ):
        return [ [ p*self.__alpha for p in ps ] for ps in probs ]

    def reweight( self, probs, data, modelprobs, modelTrainData, modelTrainWeights, voter ):
        cutoffThreshold = min( 1.0, 1000.0 / len( data ) )

        print >>sys.stderr, "    boosting"
        error = self.__getWeightedLiftAsError( modelprobs, modelTrainData, modelTrainWeights )
        #errorMargin = self.__getErrorMarginsAsP1( probs, data )
        print >>sys.stderr, "        weighted error: %f" % error

        if error > 0.5:
            print >>sys.stderr, "        oh dear, error too high"
            print >>sys.stderr, "        alpha:                    %f" % 0.0
            self.__alpha = 0.0
            return

        if error == 0.0:
            alpha = 5.0
        else:
            alpha = min( 5.0, 0.5 * math.log( (1-error)/error, 2 ) )
        print >>sys.stderr, "        alpha:                    %f" % alpha

        s = 0.0
        newWeights = []

        if self.__verbose:
            dbg = [ [0,0],[0,0] ]

        for ps, d, w in zip( probs, data, self.__weights ):
            c, feats = d

            if c == 0:
                f = ps[1]-ps[0]
            else:
                f = ps[0]-ps[1]

            #if c == 0:                                            # class=0
            #    if ps[1] >= errorMargin[0]:     # above lowest class=1
            #        f = 1.0                                         # is wrong, weight up
            #    else:
            #        f = -1.0
            #else:                                         # class=1
            #    if ps[1] < errorMargin[1]:
            #        f = 1.0
            #    else:
            #        f = -1.0
                #if ps[1] < upper20p:        # below the upper 20%
                #    f = ps[0]                         # is wrong, weight up
                #else:
                #    f = -ps[1]

            if self.__verbose:
                if f > 0:
                    dbg[c][1] += 1
                elif f < 0:
                    dbg[c][0] += 1

            newW = w * math.exp( alpha * f )
            newWeights.append( newW )
            s += newW

        self.__weights = [ w/s for w in newWeights ]

        if self.__verbose:
            dbgDescr = [ 'down', '    up' ]
            for c in (0,1):
                for s in (0,1):
                    print >>sys.stderr, "        class %2d weighted %s: %d" % ( c, dbgDescr[s], dbg[c][s] )
            print >>sys.stderr,         "        highest weight:                 %f" % ( max( self.__weights ) )
            print >>sys.stderr,         "        cutting off at:                 %f" % ( cutoffThreshold )

        modified = 0.0
        for i, w in ( (i,w) for (i,w) in enumerate( self.__weights ) if w > cutoffThreshold ):
            modified += w
            self.__weights[i] = 0.0
        if modified > 0.0:
            s -= modified
            self.__weights = [ w/s for w in newWeights ]

        self.__alpha     = alpha

    def __getErrorMarginsAsP1( self, probs, data ):
        cprobs = [ (p[1],d[0]) for p,d in izip( probs, data ) ]
        cprobs.sort( key = lambda x: x[0], reverse = True )

        top = int( len( data )*0.2 )
        upper20p = cprobs[top-1][0]
        lowestP1 = dropwhile( lambda( p, c ): c == 0,    reversed( cprobs ) ).next()[0]

        errorMarginC0 = max( lowestP1, upper20p )
        errorMarginC1 = upper20p

        return errorMarginC0, errorMarginC1

    def __getWeightedLiftAsError( self, probs, data, weights ):

        cprobs = [ (p[1],d[0],w) for p,d,w in izip( probs, data, weights ) ]
        cprobs.sort( key = itemgetter(0), reverse = True )

        top = int( len( data )*0.2 )
        #upper20p = cprobs[top-1][0]
        #lowestP1 = dropwhile( lambda( p, c, w ): c == 0,    reversed( cprobs ) ).next()[0]

        error = 0.0
        # add up weights of class=0 above lowest class=1
        above0 = sum( w for p, c, w in ifilter( lambda( p, c, w ): c == 0, dropwhile( lambda( p, c, w ): c == 0,    reversed( cprobs ) ) ) )

        # add up weights of class=0 in upper 20%
        upper0 = sum( w for p, c, w in cprobs[:top] if c == 0 )

        error += min( upper0, above0 )

        # add up weights of class=1 in lower 20%
        error += sum( w for p, c, w in cprobs[top:] if c == 1 )

        # normize
        error /= sum( weights )
        return error

        #total = sum( w for w, c in ifilter( lambda (w, c): c == 1, izip( weights, imap( lambda (c, f): c, data ) ) ) )
        #total /= sum( weights )

        #cprobs = cprobs[:top]
        #c1up = sum( w for p, c, w in ifilter( lambda (p, c, w): c == 1, cprobs ) )
        #totalup = sum( w for p, c, w in cprobs )
        #frac = c1up/totalup

        # not sure if this works...
        #error = 0.5 - max( -1.0, min( 1.0, math.log( frac/total, 5 ) ))/2
        #return error


class AdaBoost:
    def __init__( self, data, verbose = False ):
        dataCount = len( data )
        self.__weights = [ 1.0/dataCount ] * dataCount
        self.__alpha     = 1.0

    def getWeights( self ):
        return self.__weights

    def alpha( self ):
        return self.__alpha

    def weightProbs( self, probs ):
        return [ [ p*self.__alpha for p in ps ] for ps in probs ]

    def reweight( self, probs, data, modelprobs, modelTrainData, modelTrainWeights, voter ):
        print >>sys.stderr, "  boosting"
        error = self.__getWeightedError( modelprobs, modelTrainData, modelTrainWeights )
        print >>sys.stderr, "    weighted error: %f" % error

        if error > 0.5:
            print >>sys.stderr, "    oh dear, error too high"
            print >>sys.stderr, "    alpha:                    %f" % 0.0
            self.__alpha = 0.0
            return

        alpha = 0.5 * math.log( (1-error)/error, 2 )
        print >>sys.stderr, "    alpha:                    %f" % alpha

        s = 0.0
        newWeights = []
        for ps, d, w in zip( probs, data, self.__weights ):
            c, feats = d

            f = 0.0
            for k,p in enumerate(ps):
                if k == c: f -= p
                else:            f += p
                #f += boostMatrix[k][c] * p
            newW = w * math.exp( alpha * f )
            newWeights.append( newW )
            s += newW

        self.__weights = [ w/s for w in newWeights ]
        self.__alpha     = alpha

    def __getWeightedError( self, probs, data, weights ):
        errors = 0.0
        s = 0.0
        for ps, d, w in zip( probs, data, weights ):
            c, feats = d
            s += w
            for v,p in enumerate(ps):
                if v != c: errors += p * w
        return errors/s



class NoBoost:
    def __init__( self, data, verbose = False ):
        dataCount = len( data )
        self.__weights = [ 1.0/dataCount ] * dataCount

    def getWeights( self ):
        return self.__weights

    def reweight( self, probs, data, modelprobs, modelTrainData, modelTrainWeights, voter ):
        pass

    def weightProbs( self, probs ):
        return probs


