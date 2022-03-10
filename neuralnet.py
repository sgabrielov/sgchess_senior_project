# -*- coding: utf-8 -*-

import bitstring
import random
import numpy as np

# NEURAL NETWORK CONFIGURATION

# Number of nodes in each of the columns
COL_NODES = [16,4,4,1]

# Number of nodes in output column
OUT_NODES = 1

# END CONFIGURATION


class NeuralNet:
    """
    The class which represents operations for the neural network

    ...

    Attributes
    ----------


    Methods
    -------

    """
    
    node_list = []
    biases = []
    
    #def __init__(self):
        
       
        
        
    def loadinputbits(self, inbits: bitstring.BitArray):
        
        
        if(len(inbits)!=COL_NODES[0]):
            raise ValueError("Expected %d input bits, received %d" % (COL_NODES[0], len(inbits)))
            
        self.inputbits = np.array(inbits)
        
    def loadeval(self, position_eval:float = 0):
        
        self.position_eval = position_eval
    
    def processrow(self, row: list):
                
        self.loadinputbits(bitstring.BitArray(row[0][2]))
        
        self.loadeval(row[0][3])        
        
            
    def printintputbits(self):
        print(self.inputbits.bin)
    
    def printeval(self):
        print(self.position_eval)
        
    def initializeweights(self, zeroinit:bool = False):
        
        
        randmin = -1
        randmax = 1
        
        # A list of numpy arrays representing the biases of each node
        blist = []
                
        # A temporary variable that will be used to store a weights matrix, in numpy array form
        mlist = []
        
        # A list of numpy arrays representing the node values for every column
        nlist = []
             
        
        
        # Create a new numpy object that will store the weights matrix and biases array for the interactions
        # between the input layer and first hidden layer

        # Add these numpy objects to lists for storage
        
        ### FOR MEMORY SAVINGS ###
        # Load only one weights and biases matrix at a time from disk
        
        ### INPUT LAYER <-----> FIRST HIDDEN LAYER ###
        
        # mlist.append(np.empty((COL_NODES[0], IN_NODES)))
        # blist.append(np.empty(COL_NODES[0]))
        
        # print(np.shape(mlist[0]))
        # print(np.shape(blist[0]))

        
        # for i in range(COL_NODES[0]):
        #     blist[0][i] = self.getrandom()
        #     for j in range(IN_NODES):
        #         mlist[0][i][j] = self.getrandom()
                
                
        
        # ### HIDDEN LAYER <-----> HIDDEN LAYER ###
        
        # for i in range(1, NUMCOLS):
            
        #     mlist.append(np.empty((COL_NODES[i], COL_NODES[i-1])))
        #     blist.append(np.empty(COL_NODES[i]))
            
        #     for j in range(COL_NODES[i]):
        #         blist[i][j] = self.getrandom()
        #         for k in range(COL_NODES[i-1]):
        #             mlist[i][j][k] = self.getrandom()
                    
        # ### LAST HIDDEN LAYER <-----> OUTPUT LAYER ###
        
        # mlist.append(np.empty((OUT_NODES, COL_NODES[NUMCOLS-1])))
        # blist.append(np.empty(OUT_NODES))
        
        # for i in range(OUT_NODES):
        #     blist[NUMCOLS][i] = self.getrandom()
        #     for j in range(COL_NODES[NUMCOLS-1]):
        #         mlist[NUMCOLS][i][j] = self.getrandom()
        
        # Initialize all weights and bises with random values
        
        for i in range(len(COL_NODES)-1):
            mlist.append(np.empty((COL_NODES[i+1], COL_NODES[i])))
            blist.append(np.empty(COL_NODES[i+1]))
            
            for j in range(COL_NODES[i+1]):
                blist[i][j] = self.getrandom()
                for k in range(COL_NODES[i]):
                    mlist[i][j][k] = self.getrandom()
        
        
        self.node_list = mlist       
        self.biases = blist
        self.node_values = []
                
        
                
    def getrandom(self, randmin:int = -1, randmax:int = 1, zeroinit:bool = False):
        if(zeroinit):
            return 0
        else:
            return random.random() * (randmax - randmin) + randmin
    
    def printweights(self, randmin:int = -1, randmax:int = 1):
        
        # print("Hidden column 1:")
        # for i in range(COL_NODES[0]):
        #     for j in range(IN_NODES):
        #         print("HC%d   %f   IN%d" % (i+1, self.node_list[0][i][j], j+1))
        #     print("HC%d bias: %f" % (i+1, self.biases[0][i]))
        
        # for i in range(1, NUMCOLS):
        #     print("Hidden column %d" % (i+1))
        #     for j in range(COL_NODES[i]):
        #         for k in range(COL_NODES[i-1]):
        #             print("HC%d   %f   HC%d" % (j+1, self.node_list[i][j][k], k+1))
        #         print("HD%d bias: %f" % (j+1, self.biases[0][j]))
        
        # print("Output Column:")            
        # for i in range(OUT_NODES):
        #     for j in range(COL_NODES[NUMCOLS-1]):
        #         print("OC%d   %f   HC%d" % (i+1, self.node_list[NUMCOLS][i][j], j+1))
        #     print("OC%d bias: %f" % (i+1, self.biases[NUMCOLS][i]))
        
        for i in range(0, len(COL_NODES)-1):
            print("Column %d" % (i))
            for j in range(COL_NODES[i+1]):
                for k in range(COL_NODES[i]):
                    print("Node%d   %f   Node%d" % (j+1, self.node_list[i][j][k], k+1))
    
    # Processes the calculations of the columns, weights, biases of the network
    # Returns the output vector of the network    
    
    def feedforward(self, input_vector:np.ndarray, weights:np.ndarray, biases:np.ndarray):

        for w, b in zip(weights, biases):
            input_vector = self.sigmafy(np.dot(w, input_vector) + b)
        
        self.node_values.append(input_vector)
        return input_vector
    
    def expandoutput(self, outputnode:np.array):
        
        mineval = -50
        maxeval = 50
        
        return outputnode[0] * (maxeval - mineval) + mineval
        
    def calcoutput(self):
               
        self.output_value = self.expandoutput(self.feedforward(self.inputbits, self.node_list, self.biases))
        
                    
    def sigmafy(self, x:np.float64):
        
        return 1 / (1 + np.exp(0 - x))
        
    def dsigmafy(self, x:float):
        
        return (self.sigmafy(x) * (1 - self.sigmafy(x)))
    
    def dCda(self, a:float, y:float):
        
        return 2 * (a - y)
    
    def dadz(self, z:float):
        
        return self.dsigmafy(z)
    
    def dzdw(self, a:float):
        
        return a
    
    # def backpropagate(self):
        
    #     gradients = []
    #     bias_gradients = []
        
                
    #     for i in range(0, len(COL_NODES)-1):
    #         gradients.append(np.empty((COL_NODES[i+1], COL_NODES[i])))
    #         bias_gradients.append(np.empty(COL_NODES[i+1]))
                
    #     print(gradients)
        
        
    #     # for i in range(len(COL_NODES)-1, 0 ,-1):
    #     #     for j in range(COL_NODES[i-1]):
    #     #         for k in range(COL_NODES[i]):
    #     #             sumofdeltas = 0
    #     #             for in range(COL_NODES[i]):
    #     #                 delta = 
    #     #             delta = self.dCda()
    #     #             sumofdeltas += delta
    #     #             gradients[i][j][k] = delta
                
        
    #     # sumofdeltas = 0
    #     # delta = 1
    #     # for i in range(COL_NODES[NUMCOLS-1]):
    #     #     for j in range(OUT_NODES):
    #     #         delta = self.dCda(self.node_list[NUMCOLS][i][j], self.position_eval) * self.dadz(node_list[NUMCOLS-1][i])
                
    #     #     gradient_vectors[NUMCOLS-1][i] = sumofdeltas
        
        
    def loadweights(self, filename: str):
        
        a = 1
        
    def saveweights(self, filename: str):
        
        a = 1
        
    def printoutput(self):
        print(self.output_value)