#M/M/C/C queue implementation

#considered a model where the position of the server is irrelevant, just uses counters


#Data Structures and Set up------------------------------------------------------------------------------
import random
from math import factorial
import pickle
import matplotlib.pyplot as plt

#CONSTANTS
#No. of servers
c = 16
arrivalRate = 0.01
SERVICE_AVERAGE = 100
ARRIVAL_RATE_STEP = 0.001
ARRIVAL_RATE_MIN = 0.01
ARRIVAL_RATE_MAX = 0.15

#HyperParameters
NO_OF_ARRIVALS_FOR_END = 10000


#Data structures / System state variables
servers = [False]*c

#State anf global variables
simTime = 0    
timeLast = 0
nextEventType = 0
timeNextEvent = [-1]*(c+1)  #Store arrival in position C+1, all others correspond to server index
totalArrivals = 0
totalDepartures = 0
totalLoss = 0
areaServerStatus = 0    #uptime of servers - no. of servers * time, updated each timestep
areaServersStatus = [0]*(c)

#Events--------------------------------------------------------------------------------------------------

#arrive
#simulates the arrival to the queue system, updates state variables
#INPUTS - None
#OUTPUTS - None
def arrive():
    #declare global vars:
    global totalArrivals, totalLoss, servers, arrivalRate
    allocated = False
    #allocate arrival to free state var, give service time
    for i in range(len(servers)):
        if servers[i] == False:
            #allocate
            allocated = True
            break
        else:
            #check another server
            pass   

    #alter state vars accordingly
    totalArrivals += 1
    if allocated == True:
        servers[i] = True
        timeNextEvent[i] = simTime + ExpService()
    else:
        totalLoss += 1
    #generate next arrival
    timeNextEvent[16] = simTime + ExpArrival(arrivalRate)


#depart
#simulates a departure from the queuwing system, updates state variables
#INPUTS - minIndex (int) - index of departure
#OUTPUTS - None
def depart(minIndex):
    #delcare global vars
    global totalDepartures
    #free up server slot with index minIndex
    servers[minIndex] = False
    timeNextEvent[minIndex] = -1
    #set state vars
    totalDepartures += 1


#endSim
#This function compares the current no. of processed customers with the termination criteria,
#deciding whether or not to terminate the program
#INPUTS - terminationVar (int) - comparator value, terminationVal - termination value
#OUTPUTS - BOOL, True when termination should end
def endSim(terminationVar, terminationVal = NO_OF_ARRIVALS_FOR_END):
    #check if var = termination criteria
    if terminationVar >= terminationVal:
        return True
    else:
        return False


#Timing function
#This function is used to locate the next event and increment the system clock
#INPUTS - None
#OUTPUTS - minIndex (int) - Index of next event
def timing():
    #access global vars
    global simTime
    #find next event, min in next event times   
    mIndex = lambda lst: lst.index(min(x for x in lst if x >= 0))
    minIndex = mIndex(timeNextEvent)
    #update system clock
    simTime = timeNextEvent[minIndex]
    #update sim times inside 
    return minIndex
    
#Which event
#Determines the nature of the next event - arrival or departure
#INPUTS - minIndex (int) - index of next event
#OUTPUTS - nextEventType (char) - a or d
def whichEvent(minIndex):
    if minIndex == 16:
        nextEventType = "a"
    else:
        nextEventType = "d"
    return nextEventType


#resets all state variables
#as described above, resets to original values to ensure smoothness between disjoint simulations
#INPUTS - None
#OUTPUTS - None
def resetStateVars():
    global arrivalRate, servers, simTime, timeLast, nextEventType, timeNextEvent, totalArrivals, totalDepartures, totalLoss, areaServerStatus, c
    #compile into a dictionary, do calculation for uptime#
    servers = [False]*c
    #State variables
    simTime = 0    
    timeLast = 0
    nextEventType = 0
    timeNextEvent = [-1]*(c+1)  #Store arrival in position C+1, all others correspond to server index
    totalArrivals = 0
    totalDepartures = 0
    totalLoss = 0
    areaServerStatus = 0    #uptime of servers - no. of servers * time, updated each timestep
    areaServersStatus = [0]*(c)


#Update Statistics---------------------------------------------------------------------------------------


#update stats (whole simulation)
#This function calculates the areaServerStatus (%active)
#INPUTS - None
#OUTPUTS - None
def updateStats():
    global simTime, timeLast, areaServerStatus
    #calc area, no. of active servers * diff in time
    currentArea = (simTime-timeLast)*servers.count(True)
    #update timeLast
    timeLast = simTime
    areaServerStatus += currentArea
    

#update stats (individual servers)
#This function calculates the areaServerStatus (%active) for each server
#INPUTS - None
#OUTPUTS - None
def updateStatsList():
    #calc area increase for each server individually
    global simTime, timeLast, areaServersStatus
    for i in range(len(servers)):
        if i == True:
            areaServerStatus[i] += (simTime-timeLast)
    #update timeLast
    timeLast = simTime         


#Exponential functions-------------------------------------------------------------------------------------------------------


#Exponential function for service time
#Produces exponential value using the mean
#INPUT - mean (float) - mean of the distribution
#OUTPUT - exponentially distributed value (float)
def ExpService(mean = SERVICE_AVERAGE):
    return random.expovariate(1.0/mean)

#Exponential function for arrival time
#Produces exponential value using the arrival rate
#INPUT - rate (float) - arrival rate
#OUTPUT - exponentially distributed value (float)
def ExpArrival(rate):
    return random.expovariate(rate)


#Diagnostic/report functions-------------------------------------------------------------------------------------------------------


#Prints the state variables
def printStateVars():
    print(f'''
-----------------------------------------------
simTime = {simTime}
timeLast = {timeLast}
nextEventType = {nextEventType}
timeNextEvent = {timeNextEvent}
totalLoss = {totalLoss}
areaServerStatus = {areaServerStatus}
totalArrivals = {totalArrivals}
-----------------------------------------------''')

#prints the server status
def printServerStatus():
    print(f'''
-----------------------------------------------
Total servers = {c} 
No. of available servers = {(lambda x: sum(1 for item in x if not item))(servers)}
No. of unavailable servers = {(lambda x: sum(0 for item in x if not item))(servers)}
servers: {servers}
-----------------------------------------------''')


#Final report
#This function aggregates all the data from the simulation into a final report, in the form of a dictionary
#INPUT - None - accesses all global variables
#OUTPUT - dictionary of statistical data (many types)
def finalReport():
    global c, arrivalRate, servers, simTime, timeLast, nextEventType, timeNextEvent, totalArrivals, totalDepartures, totalLoss, areaServerStatus
    reportVals = {"noOfServers": c, "arrivalRate": arrivalRate, "simEndTime": simTime, "totalArrivals": totalArrivals, "totalDepartures": totalDepartures,
                  "totalDepartures": totalDepartures, "totalLoss": totalLoss, "areaServerStatus": areaServerStatus, "areaServerStatusProp": areaServerStatus/(simTime*c)}
    return reportVals



#Analytical models---------------------------------------------------------------------------------------


#calculates theoritical blocking probabilty
#calculates the blocking probability according to the Erlang-B formula
#INPUTS - arrivalRate (float) - arrival rate of simulation
#         serviceRate (float) - service rate of simulation
#                   c (int)   - no. of servers
#OUTPUTS - blocking probability (float)
def blockingProbability(arrivalRate, serviceRate, c):
    #calculate components of equation
    numerator = ((arrivalRate/serviceRate)**c)/(factorial(c))
    denominator = (sum((arrivalRate / serviceRate) ** k / factorial(k) for k in range(c + 1)))
    return numerator/denominator


#calculates the theoretical server utilisation proportion
#This implementation is applied to M/M/C queues, and doesn't consider blocking probabilty
#INPUTS - arrivalRate (float) - arrival rate of simulation
#         serviceRate (float) - service rate of simulation
#                   c (int)   - number of servers in sim
#OUTPUTS - The utilisation proportion (float)
def theoreticalUtilProportion1(arrivalRate, serviceRate, c):
    numerator = arrivalRate
    denominator = (1/serviceRate) * c
    return numerator / denominator


#calculates the theoretical server utilisation proportion
#This implementation considers blocking probability
#INPUTS - arrivalRate (float) - arrival rate of simulation
# blockingProbability (float) - chance for an arrival to be blocked
#         serviceRate (float) - service rate of simulation
#                   c (int)   - number of servers in sim
#OUTPUTS - The utilisation proportion (float)
def theoreticalUtilProportion2(arrivalRate, blockingProbability, serviceRate, c):
    numerator = arrivalRate * (1-blockingProbability)
    denominator = (1/serviceRate) * c
    return numerator / denominator


#calculates the theoretical no. of customers in the system 
#calculate the stationary distribution for each k, k in c. Multiply the sum of distributions by c,
#INPUTS - arrivalRate (float) - arrival rate
#         serviceRate (float) - 1 / service average
#                   c (int)   - number of servers
#OUTPUTS - averageCustomersInSystem (float) - float detailing number of customers in the system
def theoreticalNoOfCustomersInTheSystem(arrivalRate, serviceRate, c):
    stateProbabilities = []
    p0 = (sum(((arrivalRate/serviceRate)**k)*(1/factorial(k))  for k in range (c+1)))**-1
    stateProbabilities.append(p0)
    
    averageCustomersInSystem = sum(
        k * p0 * (arrivalRate/serviceRate)**k / factorial(k)
        for k in range(c+1)
    )

    return averageCustomersInSystem


#Main-------------------------------------------------------------------------------------------------------


#MAIN
def main():
    #define global vars
    global nextEventType, totalArrivals, totalLoss, arrivalRate, simTime
    #initialise - create first arrival
    timeNextEvent[c] = ExpArrival(arrivalRate)
    #while running
    endSimVal = False
    while endSimVal == False:
        #timing routine
        minIndex = timing()
        #decide event type
        nextEventType = whichEvent(minIndex)
        #UPDATE STATISTICS
        updateStats()
        #Event routine
        if nextEventType == "a":
            #arrival routine
            arrive()
        elif nextEventType == "d":
            #departure routine
            depart(minIndex)
        else:
            #error handling
            print("Invalid event type")
        #is simulation over?
        endSimVal = endSim(totalArrivals, NO_OF_ARRIVALS_FOR_END)
    #create report
    return finalReport()




#generates the blocking results
#This function runs the simulation a number of times, to produce data for numerous arrival rates, thus
#experimenting
#INPUT - None
#OUTPUT - Saves data to a pickle file
def createBlockingResults():
    #globals
    global totalLoss, totalArrivals, arrivalRate
    #datastructore
    blockingResults = []
    arrivalRate = 0.01
    #run the sim under different arrivalRate conditions
    while arrivalRate <= ARRIVAL_RATE_MAX:
        #data structure to hold results for current arrival rate
        tempAr = [arrivalRate]
        print(arrivalRate)
        #make comparisons
        TBProb = blockingProbability(arrivalRate, 1/SERVICE_AVERAGE, c)
        #record important data
        tempAr.append(TBProb)
        #run simulation 3 times
        for i in range (10):
            #clear simulation
            resetStateVars()
            main()
            currentBP = totalLoss / (totalArrivals)
            #record result of sim
            tempAr.append(currentBP)
        #increment arrival rate
        arrivalRate += ARRIVAL_RATE_STEP
        blockingResults.append(tempAr)
    print(blockingResults)
    with open('blockingResults2.pkl', 'wb') as file:
        pickle.dump(blockingResults, file)


#creates the server utilisation results
#runs the simulation a number of times, produces results based off of these runs, also produces the 
#theoretical expected utilisation
#INPUTS - None
#OUTPUTS - None - saves data to pickle file
def createServerUtilResults():
    #assign globals
    global arrivalRate, c, simTime
    #assign data structures
    utilResults = []
    arrivalRate = 0.01
    #initialise while loop
    while arrivalRate <= ARRIVAL_RATE_MAX:
        print(arrivalRate)
        tempAr = [arrivalRate]
        #run theoretical model
        Bprob = blockingProbability(arrivalRate, 1/SERVICE_AVERAGE, c)
        TUProp = theoreticalUtilProportion2(arrivalRate, Bprob, SERVICE_AVERAGE, c)
        TUProp2 = theoreticalUtilProportion1(arrivalRate, SERVICE_AVERAGE, c)
        tempAr.append(TUProp)
        #run simulations
        for i in range (10):
            #clear simulation
            resetStateVars()
            report = main()
            #record result of sim
            tempAr.append(report["areaServerStatusProp"])
        #increment arrivalRate
        arrivalRate += ARRIVAL_RATE_STEP
        #store results
        utilResults.append(tempAr)
    #save data for arrivalRate
    print(utilResults)
    with open('pickles\\utilResults5.pkl', 'wb') as file:
        pickle.dump(utilResults, file)
    

#Calculates the average number of customers in the siulation, produces results for examination
#runs the sim under different arrival rate conditions, numerous times, calc average no. of customers in
#system using area server status divided by simTime
#INPUTS - None
#OUTPUTS - None - saves file
def noOfCustomersResults():
    global arrivalRate, c, simTime
    #set up data structures
    custResults = []
    arrivalRate = 0.01
    #initialise while loop
    while arrivalRate <= ARRIVAL_RATE_MAX:
        print(arrivalRate)
        #data structs
        tempAr = [arrivalRate]
        #calculating values
        averNoOfCust = theoreticalNoOfCustomersInTheSystem(arrivalRate, 1/SERVICE_AVERAGE, c)
        tempAr.append(averNoOfCust)
        for i in range(10):
            #clear simulation
            resetStateVars()
            report = main()
            #record result of sim
            averageNoOfCustomersInSystem = (report["areaServerStatus"]/report["simEndTime"])
            #MIGHT NEED TO ADD ---------------------------------------------------------------------
            #+ arrivalRate/(sevice mean)
            tempAr.append(averageNoOfCustomersInSystem)
        #increment 
        arrivalRate += ARRIVAL_RATE_STEP
        #add to main array
        custResults.append(tempAr)
        #print(f"theoretical = {tempAr[0]}, \nactual = {sum(tempAr[1:])/(len(tempAr)-1)}")
    #save to file
    with open('pickles\\NoOfCustResults2.pkl', 'wb') as file:
        pickle.dump(custResults, file)


# performs one run with sampling
# Takes regular samples (every 10 arrivals), returns the BP and SU at each sample in list form
# INPUTS - None
# OUTPUTS - BPSeries - List of BP at each sample
#           SUSeries - List of SU at each sample
def welchsMethodSingle():
    # vars
    global simTime, c, nextEventType, totalArrivals, totalLoss, areaServerStatus
    arrivalRate = 0.1
    sampleRate = 10
    numSamples = 500  # Total number of samples to collect
    sampleInterval = NO_OF_ARRIVALS_FOR_END // numSamples  # Interval at which to sample

    #reset state variables
    resetStateVars()

    # initialise - create first arrival
    timeNextEvent[c] = ExpArrival(arrivalRate)
    # data structures for analysis
    BPSeries = []
    SUSeries = []

    # while running
    while not endSim(totalArrivals, NO_OF_ARRIVALS_FOR_END):
        # timing routine
        minIndex = timing()
        # decide event type
        nextEventType = whichEvent(minIndex)
        # UPDATE STATISTICS
        updateStats()
        # Event routine
        if nextEventType == "a":
            # arrival routine
            arrive()
        elif nextEventType == "d":
            # departure routine
            depart(minIndex)
        else:
            # error handling
            print("Invalid event type")

        # collect sample at defined intervals
        if totalArrivals >= sampleInterval * len(BPSeries) and totalArrivals <= NO_OF_ARRIVALS_FOR_END:
            currentBP = totalLoss / totalArrivals
            currentSU = areaServerStatus / (simTime * c)
            BPSeries.append(currentBP)
            SUSeries.append(currentSU)

    # Ensure a final sample is taken if not already
    if len(BPSeries) < numSamples:
        currentBP = totalLoss / totalArrivals
        currentSU = areaServerStatus / (simTime * c)
        BPSeries.append(currentBP)
        SUSeries.append(currentSU)

    print(f"Lengths are {len(SUSeries)} and {len(BPSeries)}")
    return BPSeries, SUSeries



# Performs lots of run, averages to perform welchs method
# Runs 50 times, average results, creates windows and finds where the BP or SU plateaus
# INPUTS - None
# OUTPUTS - Graph plot
def overallWelch ():
    #do welchs single 50 times
    BPOverall, SUOverall = welchsMethodSingle()
    for i in range(499):
        A, B = welchsMethodSingle()
        #average all results into one BPSeries and SUSeries
        for j in range(len(A)):
            BPOverall[j] += A[j]
            SUOverall[j] += B[j]
    for i in range(len(BPOverall)):
        BPOverall[i] /= 500
        SUOverall[i] /= 500
    print(SUOverall)



    # Window length is 10 samples, equivalent to 100 iterations
    window_length = 10
    BPWindows = [BPOverall[i:i + window_length] for i in range(0, len(BPOverall), window_length)]
    SUWindows = [SUOverall[i:i + window_length] for i in range(0, len(SUOverall), window_length)]

    # Analyze each window (for example, compute average)
    BPWindowAverages = [sum(window) / len(window) for window in BPWindows]
    SUWindowAverages = [sum(window) / len(window) for window in SUWindows]

    # Calculate tick labels and positions for a subset of windows
    num_windows = len(SUWindowAverages)
    # Choose an interval for labeling (e.g., every 5 windows)
    label_interval = 5
    tick_labels = [f"{i*100}-{(i+1)*100}" for i in range(0, num_windows, label_interval)]
    tick_positions = range(0, num_windows, label_interval)  # Positions for the ticks

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(SUWindowAverages, label='Average Server Utilization')
    plt.xlabel('Iteration Range')
    plt.ylabel('Average Server Utilization')
    plt.title("Welch's Method Window Analysis, 500 simulations")
    plt.xticks(tick_positions, tick_labels, rotation=45)  # Set custom ticks
    # Add grey grid lines behind the plot
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.7, zorder=0)
    plt.legend()
    plt.show()


# Finds arrivalRate such that BP < 0.01 recursively analytically
# Recursive, calls with decimal points one lower each time
# INPUTS - arrivalRate - recursive
#          step - recursive /10 each time
#          depth - tells when to stop
# OUTPUTS - 
def findOptBPAnalytic (arrivalRate, step, depth):
    global c
    #isolate bounds where it changes using analytical model
    ogArrival = arrivalRate
    BPs = []
    for i in range(10):
        BPs.append([arrivalRate, blockingProbability(arrivalRate, 1/SERVICE_AVERAGE, c)])
        arrivalRate+=step
    #locate swap 

    index = 0
    for i in range(9):
        if (BPs[i][1] < 0.01 and BPs[i+1][1] > 0.01):
            index = i
            break
    if BPs[9][1] < 0.01:
        index = 9
    if depth == 9:
        return arrivalRate
    else: 
        return findOptBPAnalytic(ogArrival+(index/(10**depth)), step/10, depth+1)





def findOptBPReal (arrivalRate, step, depth):
    global c
    #isolate bounds where it changes using analytical model
    resetStateVars()
    ogArrival = arrivalRate
    BPs = []
    for i in range(10):
        #generate
        averageBP = 0
        for i in range(100):
            resetStateVars()
            repo = main()
            currentBP = repo["totalLoss"] / repo["totalArrivals"]
            averageBP += currentBP
            

        averageBP/=100

        BPs.append([arrivalRate, averageBP])
        arrivalRate+=step
    #locate swap 

    index = 0
    for i in range(9):
        if (BPs[i][1] < 0.01 and BPs[i+1][1] > 0.01) or (BPs[i][1] < 0.01 and BPs[i+1] == None):
            index = i
            break
    '''if BPs[9][1] < 0.01:
        index = 9'''
    print(f"it broke, index chosen is {index}")
    print(f"new arrival rate is {ogArrival+(index/(10**depth))}")
    #print(f"arrival rate: {ogArrival}")
    #print(f"what is being added: {(index/(10**depth))}")

    if depth == 9:
        return arrivalRate
    else: 
        return findOptBPReal(ogArrival+(index/(10**depth)), step/10, depth+1)


print(main())