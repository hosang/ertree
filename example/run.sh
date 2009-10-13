#!/bin/sh

../ert.py -t train.jf.gz -T test.jf.gz -e 5 -s ent --leafsize 5 -c 50 --probs nobag.nofs.entropy.probs
