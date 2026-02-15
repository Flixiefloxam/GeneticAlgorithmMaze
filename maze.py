import pygame
import random
from mazelib import Maze
from mazelib.generate.Prims import Prims
from enum import Enum
import csv
import os
import sys

algorithm = 0#algorithm to use for the agent (0 = Q-learning, 1 = genetic algorithm)
numberOfAlgorithms = 2#number of algorithms implemented
readyToDisplay = False#flag to check if the training is done and the results are ready to be displayed
windowWidth = windowHeight = 1000
blockGap = 0#gap between blocks
delay = 100#delay between agent steps in milliseconds
timeSinceLastStep = 0#time since the last step
class Actions(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
mazeSize = 10#size of the maze

#Reinforcment learning variables
Q = dict()#Q table for the Q-learning algorithm
startingEpsilon = 0.9#starting epsilon for the epsilon-greedy policy
epsilon = 0.9#epsilon for the epsilon-greedy policy
alpha = 0.1#learning rate
gamma = 0.9#discount factor
epsilonDecay = 0.999#decay rate for epsilon
minEpsilon = 0.1#minimum epsilon value
actionsTaken = 0#actions taken by the agent
episodesRun = 0#episodes run by the agent
maxEpisodes = 1000#max episodes for the Q-learning algorithm
startingMaxQValue = 0#starting max Q value
startingMinQValue = 0#starting min Q value
maxActions = 2000#max actions before resetting the maze

#Genetic algorithm variables
population = list()#population of genomes
populationFitness = list()#fitness of the population
populationSize = 100#size of the population
genomeLength = 200#length of the genome
mutationRate = 0.025#mutation rate for the genetic algorithm
maxGenerations = 1000#max generations for the genetic algorithm
generation = 0#current generation of the genetic algorithm
genomeMove = 0#current move of the genome for displaying the best genome
bestGenome = list()#best genome for the genetic algorithm
elitism = 0.05#elitism rate for the genetic algorithm

#Logging variables
run = 0#current run of the maze
OUTPUT_DIR = "logs"
logging = False#flag to check if logging is enabled(disables excessive printing to the console, fancy graphics and running of both algorithms)
algorithmDone = False#flag to check if the algorithm is done

#setting up the colors
black = (0, 0, 0)
white = (200, 200, 200)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)

#defining space types
wall = 1
path = 0
start = 2
end = 3

def main():#this runs once at the start of the program
    global screen, clock, blocksize, grid, startPos, endPos, agentPos
    #setting up the maze
    m = Maze()
    m.generator = Prims(mazeSize, mazeSize)
    m.generate()
    grid = m.grid
    blocksize = (windowHeight // (len(grid[0]))) - blockGap #size of the grid's blocks
    startPos = (1, 1)
    agentPos = startPos#agent's starting position
    endPos = (len(grid[0]) - 2, len(grid) - 2)
    grid[startPos[0]][startPos[1]] = start
    grid[endPos[0]][endPos[1]] = end

    #setting up the pygame window
    pygame.init()
    screen = pygame.display.set_mode((windowWidth, windowHeight))
    clock = pygame.time.Clock()
    screen.fill(black)
    pygame.display.set_caption("Maze")
    pygame.display.flip()

def process(delta):#called every frame
    global readyToDisplay, algorithm
    if algorithm == 0:#if the algorithm is Q-learning, run the Q-learning algorithm
        runQLearning(delta)#run the Q-learning algorithm
        if episodesRun >= maxEpisodes:#if the max episodes are reached, set the ready to display flag to true
            if logging:
                global algorithmDone
                algorithmDone = True#change the algorithm done flag to true
            else:
                readyToDisplay = True
    elif algorithm == 1:#if the algorithm is genetic algorithm, run the genetic algorithm
        runGeneticAlgorithm(delta)#run the genetic algorithm
    if readyToDisplay:#if the training is done, display the results
        drawGrid()

def runGeneticAlgorithm(delta):#run the genetic algorithm
    global population, generation, maxGenerations, genomeMove, readyToDisplay, bestGenome, timeSinceLastStep, populationFitness, agentPos
    if generation < maxGenerations:#if the max generations is not reached, run the genetic algorithm
        nextGeneration = list()#next generation of genomes
        for i in range(populationSize):#for each genome in the population
            fitness, done, steps = evaluateGenome(population[i])
            populationFitness[i] = fitness
        
        if not readyToDisplay:#if the training is not done, log the generation and best fitness
            #writer.writerow([generation, max(populationFitness), done, steps])#write the generation and best fitness to the log file
            bestIdx = populationFitness.index(max(populationFitness))#get the index of the best genome
            bestFitness = populationFitness[bestIdx]#get the fitness of the best genome
            _, bestDone, bestSteps = evaluateGenome(population[bestIdx])#evaluate the best genome
            writer.writerow([generation, bestFitness, bestDone, bestSteps])#write the generation and best fitness to the log file

        #Creating the next generation
        for i in range(int(populationSize*(1-elitism))):#for each genome in the population
            parents = selectParents()#select parents for the genetic algorithm
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            nextGeneration.append(child)#add the child to the next generation

        #Elitism
        for i in range(int(populationSize*elitism)):
            bestGenome = population[populationFitness.index(max(populationFitness))]
            population.remove(bestGenome)#remove the best genome from the population
            nextGeneration.append(bestGenome)

        
        population = nextGeneration#set the population to the next generation
        populationFitness = [0] * populationSize
        generation += 1#increment the generation
        if not logging:
            print("Generation: ", generation)
    elif not readyToDisplay:#when the max generations is reached, display the results
        for i in range(populationSize):#for each genome in the population
            fitness, finished, steps = evaluateGenome(population[i])
            populationFitness[i] = fitness
        bestFitness = max(populationFitness)
        bestGenome = population[populationFitness.index(bestFitness)]#get the best genome based on fitness
        if not logging:
            readyToDisplay = True#set the ready to display flag to true
        else:
            global algorithmDone, algorithm
            algorithmDone = True#change the algorithm done flag to true
            
        reset()#reset the agent's position
        if not logging:
            print("Training done")
            print("Best fitness: ", bestFitness)
    
    if readyToDisplay:#if the training is done, display the results
        timeSinceLastStep += delta
        if timeSinceLastStep >= delay:
            timeSinceLastStep = 0
            if genomeMove < len(bestGenome):#if the genome move is less than the length of the best genome, run the best genome
                action = bestGenome[genomeMove]
                newState, done, reward = step(action)
                agentPos = newState
                print("Move: ", genomeMove)
                print("Action: ", action)
                genomeMove += 1
                if done:# if the agent has reached the end, reset the maze
                    genomeMove = 0
                    reset()
            else:#if the genome move is greater than the length of the best genome, reset the maze
                genomeMove = 0
                reset()

def runQLearning(delta):#run the Q-learning algorithm
    global timeSinceLastStep, epsilon, minEpsilon, episodesRun
    timeSinceLastStep += delta#increment the time since the last step

    if timeSinceLastStep >= delay or not readyToDisplay:#if the time since the last step is greater than the delay, take a step
        timeSinceLastStep = 0#reset the time since the last step
        global agentPos, actionsTaken
        action = choose_action(agentPos)#choose an action based on the Q table and epsilon-greedy policy
        newState, done, reward = step(action)
        Q[agentPos][action] += alpha * (reward + gamma * max(Q[newState].values()) - Q[agentPos][action])#update the Q table

        actionsTaken += 1
        if actionsTaken >= maxActions:#if the agent has taken too many actions, reset the maze
            done = True
        agentPos = newState#update the agent's position

        

        if done:# if the agent has reached the end, reset the maze
            if not readyToDisplay:
                epsilon = max(minEpsilon, epsilon * epsilonDecay)#decay epsilon
            else:
                epsilon = 0
            if not logging:
                print("Episode: ", episodesRun)
                print("Epsilon: ", epsilon)
                print("Actions taken: ", actionsTaken)
            #logging
            if not readyToDisplay:
                writer.writerow([episodesRun, actionsTaken, epsilon])#write the episode, steps, and epsilon to the log file
            episodesRun += 1#increment the episodes run
            reset()
            actionsTaken = 0#reset the action taken by the agent

def drawGrid():
    for row in range(len(grid[0])):#height (y pos)
        for col in range(len(grid)):#width (x pos)
            if (col, row) == agentPos:
                x = col * (blocksize + blockGap)
                y = row * (blocksize + blockGap)
                pygame.draw.rect(screen, blue, (x, y, blocksize, blocksize))
            elif grid[row][col] == wall:
                x = col * (blocksize + blockGap)
                y = row * (blocksize + blockGap)
                pygame.draw.rect(screen, white, (x, y, blocksize, blocksize))
            elif grid[row][col] == start:
                x = col * (blocksize + blockGap)
                y = row * (blocksize + blockGap)
                pygame.draw.rect(screen, red, (x, y, blocksize, blocksize))
            elif grid[row][col] == end:
                x = col * (blocksize + blockGap)
                y = row * (blocksize + blockGap)
                pygame.draw.rect(screen, green, (x, y, blocksize, blocksize))
            elif grid[row][col] == path:
                x = col * (blocksize + blockGap)
                y = row * (blocksize + blockGap)
                pygame.draw.rect(screen, black, (x, y, blocksize, blocksize))
    pygame.display.update()

def step(action):#take a step in the maze
    global agentPos
    newState = agentPos
    reward = 0
    done = False
    if action == Actions.up:
        newState = (agentPos[0], agentPos[1] - 1)
    elif action == Actions.down:
        newState = (agentPos[0], agentPos[1] + 1)
    elif action == Actions.left:
        newState = (agentPos[0] - 1, agentPos[1])
    elif action == Actions.right:
        newState = (agentPos[0] + 1, agentPos[1])
    
    if grid[newState[1]][newState[0]] == wall:#if the new state is a wall, set the new state to the old state and reward to -10
        newState = agentPos
        reward = -10
    elif grid[newState[1]][newState[0]] == end:#if the new state is the end, set done to true and reward to 100
        done = True
        reward = 100
    elif grid[newState[1]][newState[0]] == start:#if the new state is the start, set done to true and reward to -1
        #done = True
        reward = -1
    elif grid[newState[1]][newState[0]] == path:#if the new state is a path, set done to false and reward to -1
        reward = -1
    
    return newState, done, reward#return the new state, done, and reward

#+100 for goal, -1 for every move, -10 for hitting a wall
def choose_action(state):#choose an action based on the Q table and epsilon-greedy policy
    global Q
    if random.random() < epsilon:#exploration
        action = random.choice(list(Actions))
    else:#exploitation
        action = max(Q[state], key=Q[state].get)
    return action

def populateQTable():#populate the Q table with random values
    global Q, epsilon
    epsilon = startingEpsilon#reset epsilon
    for row in range(len(grid[0])):#height (y pos)
        for col in range(len(grid)):#width (x pos)
            Q[(col, row)] = {action: random.uniform(startingMinQValue, startingMaxQValue) for action in Actions}
    Q[startPos] = {action: 0 for action in Actions}#set the start position to 0
    Q[endPos] = {action: 0 for action in Actions}#set the end position to 0

def distance(a, b):#calculate the Manhatten distance between two points
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def populatePopulation():#populate the population with random genomes
    global population
    for _ in range(populationSize):
        genome = randomGenome()#generate a random genome
        population.append(genome)#add the genome to the population
        populationFitness.append(0)#add the fitness to the population fitness

def randomGenome():#generate a random genome for the genetic algorithm
    return [random.choice(list(Actions)) for _ in range(genomeLength)]#return a random genome(list of actions) of length genomeLength

def evaluateGenome(genome):#evaluate the genome for the genetic algorithm
    global agentPos, grid, endPos
    reset()#reset the agent's position
    steps = 0#reset the steps taken by the agent
    done = False#reset the done flag
    for action in genome:
        newState, done, reward = step(action)
        agentPos = newState#update the agent's position
        steps += 1#increment the steps taken by the agent
        if done:#if the agent has reached the end, break
            break
    distanceToEnd = distance(agentPos, endPos)#calculate the manhatten distance to the end
    if distanceToEnd == 0:
        fitness = 1000 + (genomeLength - steps)
    else:
        fitness = 1 / (distanceToEnd + 1)#calculate the fitness of the genome. Fitness is inversely proportional to the distance to the end
    return fitness, done, steps

def selectParents():#select parents for the genetic algorithm
    global population, populationFitness

    return random.choices(population, weights=populationFitness, k=2)#select two parents based on their fitness
    #return random.choices(population, weights=[genome[-1] for genome in population], k=2)#select two parents based on their fitness

def crossover(parent1, parent2):#crossover two parents to create a child genome
    cut = random.randint(0, genomeLength - 1)#select a random cut point
    child = parent1[:cut] + parent2[cut:]#create the child genome by combining the two parents
    return child

def mutate(genome):#mutate the genome for the genetic algorithm
    global mutationRate
    for i in range(len(genome)):#for each gene in the genome
        if random.random() < mutationRate:#if the random number is less than the mutation rate, mutate the gene
            genome[i] = random.choice(list(Actions))#set the gene to a random action
    return genome#return the mutated genome

def reset():
    global agentPos, actionsTaken
    agentPos = startPos#reset the agent's position to the starting position
    actionsTaken = 0#reset the action taken by the agent

def exitGame():#exit the program
    pygame.quit()
    #SystemExit
    #exit()

global writer
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))#Change working directory to the script's directory so the data files can be found(For some reason it sometimes wasn't there already for me)
main()#run main to set up the maze and pygame window
if logging:
    print("Running Q-learning algorithm...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logFile = os.path.join(OUTPUT_DIR, f"qlearning_maze{run+1}.csv")
with open(logFile, 'w', newline='') as f:
    writer = csv.writer(f)
    algorithm = 0
    writer.writerow(["Episode", "Steps", "Epsilon"])
    populateQTable()#populate the Q table with random values
    while algorithmDone == False:
        clock.tick()
        delta = clock.get_time()
        process(delta)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exitGame()
    algorithmDone = False

if logging:
    print("Running genetic algorithm...")
logFile = os.path.join(OUTPUT_DIR, f"genetic_maze{run+1}.csv")
with open(logFile, 'w', newline='') as f:
    writer = csv.writer(f)
    algorithm = 1
    writer.writerow(["Generation", "BestFitness", "Reached End", "Steps"])
    populatePopulation()#populate the population with random genomes
    while algorithmDone == False:
        clock.tick()
        delta = clock.get_time()
        process(delta)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exitGame()

exitGame()#exit the program
