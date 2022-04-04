# -*- coding: utf-8 -*-

import dbimport as db
import neuralnet as nn

import bitstring
import time
import random

import matplotlib.pyplot as plt

TEST_CHUNK = 100

def line_graph(values):
    plt.plot(values)
    plt.ylabel('Cost')
    plt.xlabel('Iter x%d' % (TEST_CHUNK))
    #plt.show()
    plt.savefig('plot.png')

def main():
    
    dbname = "test.db"
    net = nn.NeuralNet()
    net.initializeweights(False)
    
    num_rows = db.countdata(dbname)
    cumulative_error = 0
    ttime = 0
    worst_eval = 0
    worst_output = 0
    worst_cost = 0
    count = 0
    errors = []
    while(True):
        count += 1
        x = random.randint(1, num_rows)
        row = db.getrow(x+1, dbname)
        net.processrow(row)
        sttime = time.time()
        net.calcoutput()
        ttime += time.time() - sttime
        cost = (net.geteval() - net.getoutput()) * (net.geteval() - net.getoutput())
        if(cost > worst_cost):
            worst_eval = net.geteval()
            worst_output = net.getoutput()
            worst_cost = cost
        if((count)%TEST_CHUNK==0):
            print("--- %s seconds ---" % (ttime))

            ttime = 0
            
            print("Row %d" % (count), end='')
            print(" | Error: %f" % (cumulative_error/TEST_CHUNK))
            print("Sample eval: %f" % net.geteval(), end=' || ')
            print("Sample out: %f" % net.getoutput())
            print("Worst cost: %f || Eval: %f || Out: %f" % (worst_cost, worst_eval, worst_output))
            worst_cost = 0
            errors.append(cumulative_error/TEST_CHUNK)
            
            cumulative_error = 0
            
            line_graph(errors)
        cumulative_error += cost
        net.backpropagate()
    #net.printintputbits()

    

def main2():
    dbname = "test.db"
    net = nn.NeuralNet()
    net.initializeweights(False)
    a = bitstring.BitArray('0xC5')
    net.loadinputbits(a)
    net.loadeval(0)
    net.calcoutput()
    net.backpropagate()
    
def main3():
    for i in range(2, 0, -1):
        print("i: %d " % (i))
main()

    