# -*- coding: utf-8 -*-

import bitstring
import random
import cupy as np
import pickle
from numba import float32
# NEURAL NETWORK CONFIGURATION

# Number of nodes in each of the columns
COL_NODES = [808,16,1]

# Number of nodes in output column
OUT_NODES = 1

# END CONFIGURATION


class NeuralNet:

    
    node_list = []
    biases = []
    
    
    def loadinputbits(self, inbits: bitstring.BitArray):
        
        self.node_values = []
        self.z_values = []
        
        if(len(inbits)!=COL_NODES[0]):
            raise ValueError("Expected %d input bits, received %d" % (COL_NODES[0], len(inbits)))
            
        self.node_values.append(np.array(inbits))
               
    def loadeval(self, position_eval:float = 0):
        
        self.position_eval = position_eval
        if(self.position_eval > 10):
            self.position_eval = 10
        if(self.position_eval < -10):
            self.position_eval = 10
            
    def loadweights(self, filename:str):
        with open(filename, 'rb') as fp:
            self.node_list = pickle.load(fp)
                
    def loadbiases(self, filename:str):
        with open(filename, 'rb') as fp:
            self.biases = pickle.load(fp)
        
    def saveweights(self, filename:str):
        with open(filename, 'wb') as fp:
            pickle.dump(self.node_list, fp)
            print("saved to %s" % (filename))
        
    def savebiases(self, filename:str):
        with open(filename, 'wb') as fp:
            pickle.dump(self.biases, fp)
            print("saved to %s" % (filename))
        
    def processrow(self, row: list):
                
        self.loadinputbits(bitstring.BitArray(row[0][2]))
        self.loadeval(row[0][3])        
                   
    def getintputbits(self):
        return self.node_values[0].bin
    
    def geteval(self):
        return self.position_eval
        
    def initializeweights(self, zeroinit:bool = False):
          
        # A list of numpy arrays representing the biases of each node
        blist = []
                
        # A temporary variable that will be used to store a weights matrix, in numpy array form
        mlist = []
        
        for i in range(len(COL_NODES)-1):
            mlist.append(np.empty((COL_NODES[i+1], COL_NODES[i])))
            blist.append(np.empty(COL_NODES[i+1]))
            
            for j in range(COL_NODES[i+1]):
                blist[i][j] = self.getrandom()
                for k in range(COL_NODES[i]):
                    mlist[i][j][k] = self.getrandom()
        
        self.node_list = mlist       
        self.biases = blist
                          
    def getrandom(self, randmin:int = -1, randmax:int = 1, zeroinit:bool = False):
        if(zeroinit):
            return 0
        else:
            return random.random() * (randmax - randmin) + randmin
    
    def printweights(self):
        

        for i in range(0, len(COL_NODES)-1):
            print("Column %d" % (i))
            for j in range(COL_NODES[i+1]):
                for k in range(COL_NODES[i]):
                    print("Node%d   %f   Node%d" % (j+1, self.node_list[i][j][k], k+1))
    
    # Processes the calculations of the columns, weights, biases of the network
    # Returns the output vector of the network    
    
    def feedforward(self, input_vector:np.ndarray, weights:np.ndarray, biases:np.ndarray):
        
        for w, b in zip(weights, biases):
            z = self.getz(w, input_vector, b)
            input_vector = self.sigmafy(z)
            self.z_values.append(z)
            self.node_values.append(input_vector)
        
        #print(self.node_values)
        return input_vector
    
    def expandoutput(self, outputnode:np.array):
        
        mineval = -10
        maxeval = 10
        
        
        return outputnode[0] * (maxeval - mineval) + mineval
    
    def compressoutput(self, outputnode:np.array):
        mineval = -10
        maxeval = 10
        
        return (outputnode - mineval) / (maxeval - mineval)
        
    def calcoutput(self):        
              
        self.output_value = self.expandoutput(self.feedforward(self.node_values[0], self.node_list, self.biases))
        
              
    def sigmafy(self, x:np.float32):
        
        return 1 / (1 + (1/np.exp(x)))
        
    def dsigmafy(self, x:np.float32):
        
        return (self.sigmafy(x) * (1 - self.sigmafy(x)))
    
    def dCda(self, a:float, y:float):
        
        return 2 * (a - self.compressoutput(y))
    
    def dadz(self, z:float):
        
        return self.dsigmafy(z)
    
    def dzdw(self, w:float):
        
        return w
    
    def dCdw(self, next_node:float, last_node:float, y:float, w:float, b:float):
        return self.dzdw(last_node) * self.dsigmafy(next_node*w+b)*self.dCda(next_node, y)
    
    def dCdb(self, next_node:float, last_node:float, y:float, w:float, b:float):
        return self.dsigmafy(next_node*w+b)*self.dCda(next_node, y) 
    
    def getz(self, weight:np.ndarray, node:np.ndarray, bias:np.ndarray):
        return np.dot(weight, node) + bias
    
    def backpropagate(self):
        
        # List of matrices which stores the amount that each weight changes
        gradients = []
        
        # List of arrays which stores the amount that each bias changes
        bias_gradients = []
        
        # dC/da: Temp value which needs to be calculated in order to calc the gradients matrix
        dCda_values = []
        
        # Temp variable used to perform the summation of dC/da over each node in the next column
        delta = 0
        
        
        # Initialize numpy arrays to empty values and set datatype to 32 bit float
        for i in range(len(COL_NODES)-1):
            gradients.append(np.empty((COL_NODES[i+1], COL_NODES[i]), dtype=np.float32))
            dCda_values.append(np.empty((COL_NODES[i+1]), dtype=np.float32))
            bias_gradients.append(np.empty(COL_NODES[i+1], dtype=np.float32))
            
        # Iterate starting from the last column in the list of node vectors
        # Until the first column
        for i in range(len(self.node_list)-1, -1, -1):
            
            # Iterate starting from the first node in the selected vector
            # Until the last node
            for j in range(len(self.node_list[i])):
                
                # For the first iteration, dC/da is calculated based on the cost function of the output value
                if(i == len(self.node_list)-1):
                    dCda_values[i][j] = self.dCda(self.node_values[i+1][j], self.position_eval)
                
                # For each subsequent iteration, dC/da is calculated by summing a product of 
                #   dC/da from the previous row
                #   the z value from the previous row
                #   the node value from the previous row
                else:
                    
                    delta = 0
                    
                    # Iterate starting from the first node in the previously processed row
                    # Until the last node
                    for jj in range(len(gradients[i+1])):     
                        
                        # Maintain a rolling sum of products
                        delta += dCda_values[i+1][jj] * self.dadz(self.z_values[i+1][jj]) * self.node_list[i+1][jj][j]
                    
                    # Assign the sum to the dCda vector corresponding to that row
                    dCda_values[i][j] = delta
               
                # Calculate the bias gradient for the node
                bias_gradients[i][j] =  dCda_values[i][j] * self.dadz(self.z_values[i][j])  
                
                # Iterate over the next column in order to calculate gradients for the current node to
                # The selected node in that column
                for k in range(len(self.node_list[i][j])):
                    
                    # Calculate the gradient for the selected weight
                    gradients[i][j][k] = dCda_values[i][j] * self.dadz(self.z_values[i][j]) * self.node_values[i][k]
            
            # Apply the gradients to the weights and biases
            self.node_list[i] -= gradients[i]
            self.biases[i] -= bias_gradients[i]
       
    def getoutput(self):
        return self.output_value