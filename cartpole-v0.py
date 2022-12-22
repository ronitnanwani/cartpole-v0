# IMPORTS
from abc import ABC
import gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make('CartPole-v0')

class NeuralNet:
    # """
    # Neural network to optimize the cartpole environment 
    # """
    def __init__(self, input_dim, hidden_dim, output_dim, test_run):
        self.input_dim = input_dim                                  #input_dim=4
        self.hidden_dim = hidden_dim                                #hidden_dim=2
        self.output_dim = output_dim                                #output_dim=2
        self.test_run = test_run                                    

    #helper functions
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def init_weights(self):
        input_weight = []
        input_bias = []

        hidden_weight = []
        out_weight = []

        input_nodes = 4

        for i in range(self.test_run):
            inp_w = np.random.rand(self.input_dim, input_nodes)
            input_weight.append(inp_w)                                                              # 15*4*4
            inp_b = np.random.rand((input_nodes))
            input_bias.append(inp_b)                                                                # 15*4
            hid_w = np.random.rand(input_nodes, self.hidden_dim)
            hidden_weight.append(hid_w)                                                             # 15*4*2
            out_w = np.random.rand(self.hidden_dim, self.output_dim)
            out_weight.append(out_w)                                                                # 15*2*2
        # out_weight=np.array(out_weight)
        # print(out_weight.shape)
        return [input_weight, input_bias, hidden_weight, out_weight]
        # print(np.array([input_weight, input_bias, hidden_weight, out_weight]).shape)

    def forward_prop(self, obs, input_w, input_b, hidden_w, out_w):

        obs = obs/max(np.max(np.linalg.norm(obs)), 1)
        Ain = self.relu(obs@input_w + input_b.T)
        Ahid = self.relu(Ain@hidden_w)
        Zout = Ahid @ out_w
        A_out = self.relu(Zout)
        output = self.softmax(A_out)

        return np.argmax(output)

    def run_environment(self, input_w, input_b, hidden_w, out_w):
        obs = env.reset()
        score = 0
        time_steps = 300
        for i in range(time_steps):
            action = self.forward_prop(obs, input_w, input_b, hidden_w, out_w)
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        return score

    def run_test(self):
        generation = self.init_weights()
        # generation=np.array(generation)
        # print(generation.shape)                                                     #generation=4*15
        input_w, input_b, hidden_w, out_w = generation
        scores = []
        # print(self.test_run)
        for ep in range(self.test_run):
            score = self.run_environment(
                input_w[ep], input_b[ep], hidden_w[ep], out_w[ep])
            scores.append(score)
        return [generation, scores]


class GA:
    # """
    # Training neural net using genetic algorithm
    # """
    def __init__(self, init_weight_list, init_fitness_list, number_of_generation, pop_size, learner, mutation_rate=0.5):
        #initilize different parameters of the GA
        self.number_of_generation = number_of_generation
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.current_generation = init_weight_list
        self.current_fitness = init_fitness_list
        self.best_gen = []
        self.best_fitness = -1000
        self.fitness_list = []
        self.learner = learner

    def generatelr(self,lmin,rmax):
        l=random.randint(lmin,rmax)
        r=random.randint(lmin,rmax)
        while(l==r):
            l=random.randint(lmin,rmax)
        if(l>r):
            l,r=r,l
        return l,r

    def helper(self,DNA_list1,DNA_list2,alpha):
        
        arr1=np.concatenate((np.concatenate((DNA_list1[0:alpha[0]],DNA_list2[alpha[0]:alpha[1]+1])),DNA_list1[alpha[1]+1:32]))
        
        arr2=np.concatenate((np.concatenate((DNA_list2[0:alpha[0]],DNA_list1[alpha[0]:alpha[1]+1])),DNA_list2[alpha[1]+1:32]))
        return arr1,arr2
        

    def crossover(self, DNA_list):
        # """
        # Generate number of offsprings from parents in DNA_list such that pop_size remains same. 
        # Think of an optimal crossover strategy
        # """
        
        newDNAs = []
        
        while(True):
            
            alpha=self.generatelr(0,15)
            arr1,arr2=self.helper(DNA_list[0],DNA_list[1],alpha)
            
            alpha=self.generatelr(16,19)
            arr1,arr2=self.helper(arr1,arr2,alpha)
            
            alpha=self.generatelr(20,27)
            arr1,arr2=self.helper(arr1,arr2,alpha)
            alpha=self.generatelr(28,31)
            arr1,arr2=self.helper(arr1,arr2,alpha)
            
            if(len(newDNAs)==self.population_size-3):
                newDNAs.append(arr1)
                break
            if(len(newDNAs)==self.population_size-2):
                break
            newDNAs.append(arr1)
            newDNAs.append(arr2)
            
        return newDNAs

    def mutation(self, DNA):
        # """
        # Mutate DNA. Use mutation_rate to determine the mutation probability. 
        # Make changes in the DNA.
        # """

        lst=[]
        x=int(self.mutation_rate * (self.population_size))
        while(len(lst)!=x):
            num=random.randint(0,14)
            if(not(num in lst)):
                lst.append(num)
                
        for i in lst:
            for j in range (3):
                DNA[i][random.randint(0,15)]=random.random()
            for j in range (1):
                DNA[i][random.randint(16,19)]=random.random()
            for j in range (2):
                DNA[i][random.randint(20,27)]=random.random()
            for j in range (1):
                DNA[i][random.randint(28,31)]=random.random()

        return DNA

    def next_generation(self):
        # """
        # Forms next generation from current generation.
        # Before writing this function think of an appropriate representation of an individual in the population.
        # Suggested method: Convert it into a 1-D array/list. This conversion is done for you in this function. Feel free to use any other method.
        # Steps
        # 1. Crossover
        # Suggested Method: select top two individuals with max fitness. generate remaining offsprings using these two individuals only.
        # 2. Mutation:
        # """
        index_good_fitness = [] #index of parents selected for crossover.
        #fill the list.
        sortedlist=sorted(self.current_fitness,reverse=True)
        index_good_fitness.append(self.current_fitness.index(sortedlist[0]))
        index_good_fitness.append(self.current_fitness.index(sortedlist[1]))

        new_DNA_list = []
        new_fitness_list = []

        DNA_list = []
        for index in index_good_fitness:
            w1 = self.current_generation[0][index]
            dna_in_w = w1.reshape(w1.shape[1], -1)
            # print(dna_in_w.shape)
            b1 = self.current_generation[1][index]
            dna_b1 = np.append(dna_in_w, b1)
            # print(dna_b1.shape)
            w2 = self.current_generation[2][index]
            dna_whid = w2.reshape(w2.shape[1], -1)
            dna_w2 = np.append(dna_b1, dna_whid)
            # print(dna_w2.shape)
            wh = self.current_generation[3][index]
            dna = np.append(dna_w2, wh)
            DNA_list.append(dna)
            # print(np.array(DNA_list).shape)
        #parents selected for crossover moves to next generation
        
        new_DNA_list += DNA_list
        
        new_DNA_list += self.crossover(DNA_list)

        #mutate the new_DNA_list
        new_DNA_list=self.mutation(new_DNA_list)

        #converting 1D representation of individual back to original (required for forward pass of neural network)
        new_input_weight = []
        new_input_bias = []
        new_hidden_weight = []
        new_output_weight = []

        for newdna in new_DNA_list:

            newdna_in_w1 = np.array(
                newdna[:self.current_generation[0][0].size])
            new_in_w = np.reshape(
                newdna_in_w1, (-1, self.current_generation[0][0].shape[1]))
            new_input_weight.append(new_in_w)

            new_in_b = np.array(
                [newdna[newdna_in_w1.size:newdna_in_w1.size+self.current_generation[1][0].size]]).T  # bias
            new_input_bias.append(new_in_b)

            sh = newdna_in_w1.size + new_in_b.size
            newdna_in_w2 = np.array(
                [newdna[sh:sh+self.current_generation[2][0].size]])
            new_hid_w = np.reshape(
                newdna_in_w2, (-1, self.current_generation[2][0].shape[1]))
            new_hidden_weight.append(new_hid_w)

            sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
            new_out_w = np.array([newdna[sl:]]).T
            new_out_w = np.reshape(
                new_out_w, (-1, self.current_generation[3][0].shape[1]))
            new_output_weight.append(new_out_w)

            #evaluate fitness of new individual and add to new_fitness_list.
            #check run_environment function for details.
            new_fitness_list.append(self.learner.run_environment(new_in_w,new_in_b,new_hid_w,new_out_w))
        new_generation = [new_input_weight, new_input_bias,
                          new_hidden_weight, new_output_weight]
        return new_generation, new_fitness_list

    def show_fitness_graph(self):
        # """
        # Show the fitness graph
        # Use fitness_list to plot the graph
        # """
        #plot
        x=list(range(1,self.number_of_generation+1))
        plt.plot(x,self.fitness_list)
        plt.xlabel('Generation')
        plt.ylabel('Max fitness Value')
        plt.title('Graph')
        plt.show()

    def evolve(self):
        # """
        # Evolve the population
        # Steps
        # 1. Iterate for number_of_generation and generate new population
        for i in range(self.number_of_generation):
            new_generation, new_fitness_list=self.next_generation()
        # 2. Find maximum fitness of an individual in this generation and update best_fitness
            
            max_fitness=max(new_fitness_list)
            
            idx=0
            if(max_fitness>=self.best_fitness):
                self.best_fitness=max_fitness
                idx=new_fitness_list.index(max_fitness)
                self.best_gen=[new_generation[0][idx],new_generation[1][idx],new_generation[2][idx],new_generation[3][idx]]
        # 3. Append max_fitness to fitness_list
            self.current_generation=new_generation
            self.current_fitness=new_fitness_list
            self.fitness_list.append(max_fitness)
        # 4. Plot the fitness graph at end. Use show_fitnes_graph()
        # """
        self.show_fitness_graph()
        #evolve
        return self.best_gen, self.best_fitness


def trainer():
    pop_size = 15
    num_of_generation = 100
    # print(env.observation_space.shape)
    # print(env.action_space)
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, pop_size)                #env.observation_space.shape=(4,)   #env.action_space.n=2
    init_weight_list, init_fitness_list = learner.run_test()
    #init_fitness_list=np.array(init_fitness_list)
    #init_weight_list=np.array(init_weight_list)
    #print(init_weight_list.shape)                                                       #init_weight_list.shape=(4,15)      init_fitness_list.shape=(15,)
    #print(init_fitness_list.shape)
    #instantiate the GA optimizer
    abc=GA(init_weight_list,init_fitness_list,num_of_generation,pop_size,learner)
    #call evolve function to obtain optimized weights.
    ans,fitnessofans=abc.evolve()
    #return optimized weights
    return ans

def test_run_env(params):
    input_w, input_b, hidden_w, out_w = params
    obs = env.reset()
    score = 0
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, 15)
    for t in range(5000):
        env.render()
        action = learner.forward_prop(obs, input_w, input_b, hidden_w, out_w)
        obs, reward, done, info = env.step(action)
        score += reward
        print(f"time: {t}, fitness: {score}")
        if done:
            break
    print(f"Final score: {score}")

def main():
    params = trainer()
    test_run_env(params)


if(__name__ == "__main__"):
    main()