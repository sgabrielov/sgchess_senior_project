# -*- coding: utf-8 -*-

import dbimport as db
import neuralnet as nn

import bitstring

TEST_CHUNK = 10000

def main():
    
    dbname = "test.db"
    net = nn.NeuralNet()
    net.initializeweights(False)
    
    num_rows = db.countdata(dbname)
    cumulative_error = 0
    for x in range(num_rows):
        
        row = db.getrow(x+1, dbname)
        net.processrow(row)
        net.calcoutput()
        if((x+1)%TEST_CHUNK==1) :
            print("Row %d" % (x+1), end='')
            print(" | Error: %f" % (cumulative_error/TEST_CHUNK))
            print("Sample eval: %f" % net.geteval(), end='')
            print("Sample out: %f" % net.getoutput())
            cumulative_error = 0
        cumulative_error += (net.geteval() - net.getoutput()) * (net.geteval() - net.getoutput())
        net.backpropagate()
    #net.printintputbits()

    

def main2():
    dbname = "test.db"
    net = nn.NeuralNet()
    net.initializeweights(False)
    a = bitstring.BitArray('0x0000')
    net.loadinputbits(a)
    net.loadeval(0.0)
    net.calcoutput()
    net.backpropagate()
    
def main3():
    for i in range(2, 0, -1):
        print("i: %d " % (i))
main()

    