# -*- coding: utf-8 -*-

import dbimport as db
import neuralnet as nn

import bitstring
import time
import random

import matplotlib.pyplot as plt

TEST_CHUNK = 10

def line_graph(values, filename):
    plt.clf()
    plt.plot(values)
    plt.ylabel('Cost')
    plt.xlabel('Iter x%d' % (TEST_CHUNK))
    #plt.show()
    plt.savefig(filename)

def save(net:nn.NeuralNet):
    net.savebiases("bias.p")
    net.saveweights("weights.p")
    
def load(net:nn.NeuralNet):
    net.loadbiases("bias.p")
    net.loadweights("weights.p")   
    
def main():
    
    dbname = "test.db"
    net = nn.NeuralNet()
    net.initializeweights(False)
    
    num_rows = db.countdata(dbname)
    num_rows = 500
    cumulative_error = 0
    worst_eval = 0
    worst_output = 0
    worst_cost = 0
    count = 0
    count2 = 0
    errors = []
    worsts = []
    
    iterlist = list(range(num_rows))
    
    
    sttime = time.time()
    ttaken = 0
    tremain = 0
    progress = 0
    epochs = 1
    for i in range(epochs):
        print("############## NEXT ITER (C%d) ##############" % (count))
        count = 0
        count2 = 0
        random.shuffle(iterlist)
        for x in iterlist:
        
            
            row = db.getrow(x+1, dbname)
            net.processrow(row)
            count2 += 1
            
            if(net.geteval()<10 and net.geteval()>-10):
                count += 1
                
                net.calcoutput()

                cost = (net.geteval() - net.getoutput()) * (net.geteval() - net.getoutput())
                if(cost > worst_cost):
                    worst_eval = net.geteval()
                    worst_output = net.getoutput()
                    worst_cost = cost
                if((count)%TEST_CHUNK==0):
                    save(net)
        
                    progress = float((i+1) * count2) / (epochs * num_rows)
                    ttaken = time.time() - sttime
                    if(progress > 0):
                        tremain = ttaken * (1-progress) / progress
                    
                    print("Progress: %.2f%%" % (progress*100))
                    print(progress)
                    if(tremain > 3600):
                        print("Time remaining: %d hours" % (tremain / 3600))
                    else:
                        print("Time remaining: %d seconds" % (tremain))
                    print("Row %d | %d" % (count, count2), end='')
                    print(" | Error: %f" % (cumulative_error/TEST_CHUNK))
                    print("Sample eval: %f" % net.geteval(), end=' || ')
                    print("Sample out: %f" % net.getoutput())
                    print("Worst cost: %f || Eval: %f || Out: %f" % (worst_cost, worst_eval, worst_output))
                    print("#############################################")
                    
                    worsts.append(worst_cost)
                    errors.append(cumulative_error/TEST_CHUNK)
                    
                    worst_cost = 0
                    cumulative_error = 0
                    
                    line_graph(errors, "error_plot.png")
                    line_graph(worsts, "worst_plot.png")
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
    net = nn.NeuralNet()
    net.initializeweights(False)
    
    cumulative_error = 0
    ttime = 0
    worst_eval = 0
    worst_output = 0
    worst_cost = 0
    count = 0
    
    errors = []
    worsts = []
    for j in range(50000):
        count += 1
        n = random.randint(0,65535)
        t = '{0:016b}'.format(n)
        b = bitstring.BitArray(bin=t)
        net.loadinputbits(b)
        net.loadeval(n)
        net.calcoutput()
        
        cost = (net.geteval() - net.getoutput()) * (net.geteval() - net.getoutput())
        
        if(cost > worst_cost):
            worst_eval = net.geteval()
            worst_output = net.getoutput()
            worst_cost = cost
        if((count)%TEST_CHUNK==0):
        
            print("Row %d" % (count), end='')
            print(" | Error: %f" % (cumulative_error/TEST_CHUNK))
            print("Sample eval: %f" % net.geteval(), end=' || ')
            print("Sample out: %f" % net.getoutput())
            print("Worst cost: %f || Eval: %f || Out: %f" % (worst_cost, worst_eval, worst_output))
            
            worsts.append(worst_cost)
            errors.append(cumulative_error/TEST_CHUNK)
            
            worst_cost = 0
            cumulative_error = 0

            line_graph(errors, "error_plot.png")
            line_graph(worsts, "worst_plot.png")            

        cumulative_error += cost
        net.backpropagate()
    print(net.getoutput())
    print(net.geteval())

main()

    