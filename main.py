# -*- coding: utf-8 -*-

import dbimport as db
import neuralnet as nn

import cupy as np

import bitstring
import time
import random



import matplotlib.pyplot as plt

TEST_CHUNK = 1000000
DB_BATCH_SIZE = 100000
TEST_SAMPLE_SIZE = 1000000

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
    
def formattime(t):
    time = t
    if time>=86400:
        print("%d days " % (time/86400), end='')
        time = time % 86400
    if time>=3600:
        print("%d hours " % (time/3600), end='')
        time = time % 3600
    if time>=60:
        print("%d minutes " % (time/60), end='')
        time = time % 60
    print("%d seconds." % (time))
    
def main():
    
    np.cuda.Device(0).use()
    dbname = "test.db"
    net = nn.NeuralNet()
    #net.initializeweights(False)
    load(net)
    
    num_rows = db.countdata(dbname)
    cumulative_error = 0
    worst_eval = 0
    worst_output = 0
    worst_cost = 0
    count = 0
    count2 = 0
    errors = []
    worsts = []
    
    #iterlist = list(range(num_rows))
    
    
    sttime = time.time()
    ttaken = 0
    tremain = 0
    progress = 0
    epochs = 6
    
    ffstart = 0
    ffend = 0
    fftotal=0
    bpstart=0
    bpend=0
    bptotal=0
    batchctr = DB_BATCH_SIZE
    
    for i in range(epochs):
        print("############## NEXT ITER (C%d) ##############" % (count))
        count = 0
        count2 = 0
        #random.shuffle(iterlist)
        rows_to_process = num_rows - TEST_SAMPLE_SIZE - DB_BATCH_SIZE
        cumulative_error = 0
        for x in range(rows_to_process):
            if(batchctr>=DB_BATCH_SIZE):
                batchctr=0
                data = db.getrowbatch(x, DB_BATCH_SIZE, dbname)
                random.shuffle(data)
        
            
            
            net.processrow(data[batchctr])
            batchctr += 1
            count2 += 1
            if(net.geteval()<10 and net.geteval()>-10):
                count += 1
                ffstart = time.time()
                net.calcoutput()
                ffend = time.time()
                fftotal += ffend - ffstart
                
                bpstart=time.time()
                net.backpropagate()
                bpend=time.time()
                bptotal += bpend - bpstart
    
                cost = (net.geteval() - net.getoutput()) * (net.geteval() - net.getoutput())
                if(cost > worst_cost):
                    worst_eval = net.geteval()
                    worst_output = net.getoutput()
                    worst_cost = cost
                if((count)%TEST_CHUNK==0):
                    save(net)
        
                    progress = float((i * rows_to_process) + count2) / (epochs * rows_to_process)
                    ttaken = time.time() - sttime
                    if(progress > 0):
                        tremain = ttaken * (1-progress) / progress
                    
                    print("Progress: %.2f%%" % (progress*100))
                    formattime(tremain)
                    print("Row %d" % (count), end='')
                    print(" | Error: %f" % (cumulative_error/TEST_CHUNK))
                    print("Sample eval: %f" % net.geteval(), end=' || ')
                    print("Sample out: %f" % net.getoutput())
                    print("Worst cost: %f || Eval: %f || Out: %f" % (worst_cost, worst_eval, worst_output))
                    print("FFTime:\t%f" % (fftotal))
                    print("BPtime:\t%f" % (bptotal))
                    print("#############################################")
                    
                    worsts.append(np.asnumpy(worst_cost))
                    errors.append((np.asnumpy(cumulative_error)/TEST_CHUNK))
                    
                    line_graph(errors, "error_plot.png")
                    line_graph(worsts, "worst_plot.png")
                    
                    worst_cost = 0
                    cumulative_error = 0
                    
                    ffstart = 0
                    ffend = 0
                    fftotal=0
                    bpstart=0
                    bpend=0
                    bptotal=0
                        
                cumulative_error += cost
                   
                
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
        n = random.randint(0,255)
        t = '{0:08b}'.format(n)
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
            
            
            worst_cost = 0
            cumulative_error = 0
          

        cumulative_error += cost
        net.backpropagate()
    print(net.getoutput())
    print(net.geteval())

if __name__ == '__main__':
    main()

    