# -*- coding: utf-8 -*-

import dbimport as db
import neuralnet as nn

import bitstring

def main():
    
    row = db.getrow(1, "test.db")
    
    net = nn.NeuralNet()
    #net.processrow(row)
    #net.printintputbits()
    #net.printeval()
    
    a = bitstring.BitArray('0x0000')
    net.loadinputbits(a)
    net.loadeval(0.0)
    net.initializeweights(False)
    net.printweights()
    net.calcoutput()
    net.printoutput()
    net.printeval()
    
    
main()

    