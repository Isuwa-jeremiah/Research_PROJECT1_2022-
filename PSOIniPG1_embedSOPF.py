import sklearn as sk
import numpy as np
import random
import csv
from sklearn.neighbors import KNeighborsClassifier

datafile = 'ionosphere.data'
class_identifier = 34
percent_training = 70
number_of_iteration = 100   
percent_of_small_feature = 30
population_size = 30
number_of_feature = 34  # 13 for wine, 34 for ionosphere,... 

#%% defining fitness function and particle class
def fitness(x,datafile,class_identifier,percent_training=70,repeat=30):
    # importing data
    with open(datafile, newline='') as csvfile: # load as string to check for classifiers
        data_raw = list(csv.reader(csvfile))
    ncol_data = len(data_raw[0]) 
    nrow_data = len(data_raw)
    target = []
    for i in range(nrow_data):
        target.append(data_raw[i][class_identifier])

    feature_columns = np.array(range(ncol_data))
    feature_columns = np.delete(feature_columns,class_identifier,0)
    data = np.loadtxt(open(datafile, "rb"), delimiter=",",usecols=feature_columns) # load as numeric
    
    nfeature = data.shape[1]
    ndata = data.shape[0]
    ntraining = round(ndata*percent_training/100)
    ntest = ndata-ntraining

    active_feature = np.array([i for i, y in enumerate(x>=0.6) if y])

    if(len(active_feature)==0):
        fit = 1
        return fit
    #print("active")
    #print(active_feature)
    # define training data and target
    fitness = []
    for j in range(repeat):
        # define training data and target
        training_index = random.sample(range(ndata),ntraining)
        training_index = np.array(training_index)
        training_data = data[training_index, : ]
        training_data = training_data[:,active_feature] # remove class identifier
        training_target = []
        for i in training_index:
            training_target.append(target[i])

        # define test data and target
        test_index =  list(set(range(ndata)).symmetric_difference(set(training_index)))
        test_index = np.array(test_index) # convert it for easy access
        test_data = data[test_index,:]
        test_data = test_data[:,active_feature] # get active features only
        test_target = []
        for i in test_index:
            test_target.append(target[i])

        feature_test = KNeighborsClassifier(n_neighbors=5)
        feature_test.fit(training_data,training_target)
        fitness.append (1.0-feature_test.score(test_data,test_target))
    mean_fitness = np.average(fitness)
    return mean_fitness

def update_position(particle,datafile,class_identifier,percent_training=70):
    particle.x = particle.x + particle.v
    particle.fitness = fitness(particle.x,datafile,class_identifier,percent_training)
    return particle
    
def update_velocity(particle,gbest,vmax = 6):
    W = 0.7298
    c1 = 1.149618
    c2 = 1.149618
    particle.v = ( W*particle.v + # inertia term
        c1*random.random()*(particle.pbest-particle.x) + # cognitive term
        c2*random.random()*(gbest-particle.x) # social term
        )
    for i in range(particle.v.shape[0]):
        if abs(particle.v[i])>vmax:
            particle.v[i] = vmax*particle.v[i]/abs(particle.v[i])
    return particle

def update_pbest(particle):
    nactive = 0
    nactive_pbest = 0
    nfeature = particle.x.shape[0]
    for i in range(nfeature):  # count active features in x and pbest
        if particle.x[i]>=0.6: # 0.6 is the threshold for active feature
            nactive = nactive+1
        if particle.pbest[i]>=0.6:
            nactive_pbest = nactive_pbest + 1
    if particle.fitness < particle.pbest_fit and nactive <= nactive_pbest: # line 6 of algorithm --- redundant?
        particle.pbest = particle.x
        particle.pbest_fit = particle.fitness
    elif 0.95*particle.fitness < particle.pbest_fit and nactive < nactive_pbest: # line 9
        particle.pbest = particle.x
        particle.pbest_fit = particle.fitness
    return particle

def update_gbest(pop,gbest,gbest_fit):
    pop_size = len(pop)
    nfeature = pop[0].x.shape[0]
    for i in range(pop_size):
        nactive_pbest = 0
        nactive_gbest = 0
        for j in range(nfeature):  # count active features in pbest and gbest
            if pop[i].pbest[j]>=0.6: # 0.6 is the threshold for active feature
                nactive_pbest = nactive_pbest+1
            if gbest[j]>=0.6:
                nactive_gbest = nactive_gbest + 1
        if pop[i].pbest_fit < gbest_fit and nactive_pbest <= nactive_gbest: # line 12-14
           #print("update gbest")
            gbest = pop[i].pbest
            gbest_fit = pop[i].pbest_fit
        elif 0.95*pop[i].pbest_fit < gbest_fit and nactive_pbest < nactive_gbest: # line 15-17
            #print("update gbest")
            gbest = pop[i].pbest
            gbest_fit = pop[i].pbest_fit
    return gbest,gbest_fit

class Particle:
    def __init__(self, variable_count,active_feature_count,fitness_function,datafile,class_identifier,percent_training=70):
        self.variable_count = variable_count
        self.x = 0.6*np.ones(self.variable_count) # particle position
        feature_id_to_activate = random.sample(range(variable_count),active_feature_count)
        self.x[feature_id_to_activate] = 0.6+random.random()*0.4
        vlist = []
        for i in range(variable_count):
            vlist.append( (random.random()-  self.x[i])/2 ) # see eq. 2
        self.v = np.array(vlist) # particle velocity
        self.pbest = self.x # particle best position
        self.fitness = fitness_function(self.x,datafile,class_identifier,percent_training)
        self.pbest_fit = self.fitness

#%% initialize population following mixed/hybrid rule
def initialize_pop(population_size,variable_count,small_feature_percent,class_identifier,percent_training=70):
    pop = []
    n_small = round(small_feature_percent/100*population_size) # number of individual with low feature active
    n_large = population_size-n_small # number of individual with many features
    n_active_small = round(0.12 * variable_count) # number of active feature for small individual
    for i in range(n_small):
        pop.append(Particle(variable_count,n_active_small,fitness,datafile,class_identifier,percent_training))  # add small individual to the population
    minimum_feature_large = round(variable_count/2) # minimum number of active feature for large individual
    for i in range(n_large):
        n_active = random.randint(minimum_feature_large,variable_count)
        pop.append(Particle(variable_count,n_active,fitness,datafile,class_identifier,percent_training))  # add large individual to the population
    return pop


def get_pop_fitness(pop):
    fitness_list = np.zeros(len(pop))
    for i in range(len(pop)):
        fitness_list[i]= pop[i].fitness
    return fitness_list

#%% test this
p = initialize_pop(population_size,number_of_feature,percent_of_small_feature,class_identifier) # input: pop size, n_feature, percent of population with small feature, classifier index
p_fitness = get_pop_fitness(p)  # example on how to evaluate the whole population at once

#%% 
def runPSO(number_of_iteration):
    gbest_id = p_fitness.argmax()
    gbest = p[gbest_id].x # gbest only store position
    gbest_fit = max(p_fitness)
    nparticle = p_fitness.shape[0]
    for i in range(number_of_iteration):
        print("iteration number")
        print(i)
        print("running PSO updates")
        for j in range(nparticle):
            p[j]=update_pbest(p[j])
            gbest,gbest_fit=update_gbest(p,gbest,gbest_fit)
        for j in range(nparticle):
            p[j]=update_velocity(p[j],gbest)
            p[j]=update_position(p[j],datafile,class_identifier,percent_training)
        nfeature = gbest.shape[0]
        
        ## SOPF applied
        print("running SOPF")
        print("running on particle ID=")
        for k in range(nparticle):
            print(k)
            for j in range(nfeature):
                temp_x = p[k].x
                if temp_x[j]>=0.6:
                    temp_x[j]=temp_x[j]-0.6
                else:
                    temp_x[j]=temp_x[j]+0.6
                temp_fitness = fitness(np.array(temp_x),datafile,
                class_identifier, percent_training)
                if(temp_fitness< p[k].fitness):
                    p[k].x = temp_x
                    p[k].fitness = temp_fitness
    return gbest,gbest_fit
    
gbest,gbest_fit = runPSO(number_of_iteration)
print("Error rate:")
print(gbest_fit)
print("Accuracy:")
print(1-gbest_fit)
nfeature = gbest.shape[0]
nactive_gbest = 0
active_feature = []
for j in range(nfeature):  # count active features in pbest and gbest
    if gbest[j]>=0.6:
        nactive_gbest = nactive_gbest + 1
        active_feature.append(1)
    else:
        active_feature.append(0)
print("Active features")
print(gbest)
print("Number of active features:")
print(nactive_gbest)

