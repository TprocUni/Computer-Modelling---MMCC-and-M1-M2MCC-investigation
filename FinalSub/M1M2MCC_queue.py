import random
from math import factorial, sqrt
import pickle
import matplotlib.pyplot as plt
import numpy as np

class QueueSimulation:
    def __init__(self, c, threshold, arrivalRateM1, arrivalRateM2, serviceAverage, arrivalsForTerm):
        self.c = c
        self.threshold = threshold
        self.arrivalRateM1 = arrivalRateM1
        self.arrivalRateM2 = arrivalRateM2
        self.serviceAverage = serviceAverage
        self.arrivalsForTerm = arrivalsForTerm

        self.resetStateVars()

    def resetStateVars(self):
        """
    Name: resetStateVars
    Description: Resets the state variables of the simulation. It initializes the status of servers, 
                 simulation times, event times, and counters for arrivals, departures, and losses for both methods.
                 
    Inputs: None (Uses the object's current state)
    Outputs: None (Modifies the object's state variables)

    """
        self.servers = [False] * self.c
        self.simTime = 0
        self.timeLast = 0
        self.timeNextEvent = [-1] * (self.c + 2) #timeNextEvent[c+1] = newcall, timeNextEvent[c+2] = handoverSlot
        self.totalArrivalsM1 = 0
        self.totalArrivalsM2 = 0
        self.totalDepartures = 0
        self.totalLossM1 = 0
        self.totalLossM2 = 0
        self.areaServerStatus = [0] * self.c #can use this approach for coverage of each type


    def expService(self):
        return random.expovariate(1.0 / self.serviceAverage)


    def expArrival(self, arrivalRate):
        return random.expovariate(arrivalRate)


    #new call
    def arrivalM1(self):
        """
    Name: arrivalM1
    Description: Handles the arrival event for Method 1 in the simulation. It checks for free servers,
                 processes an arrival by either accepting or rejecting it based on the availability 
                 and the defined threshold, updates the simulation state accordingly, and schedules the next arrival.

    Inputs: None (Uses the object's current state and methods)
    Outputs: None (Modifies the object's state variables and schedules next events)

    - Updates self.totalArrivalsM1: Increments the count of total arrivals for Method 1.
    - Determines free servers: Identifies and counts servers that are currently free (idle).
    - Accepts or rejects arrivals: Based on the availability of free servers and a predefined threshold.
    - Updates self.servers: Marks a server as busy if an arrival is accepted.
    - Schedules next service event: Sets the time for the next service event if an arrival is accepted.
    - Updates self.totalLossM1: Increments the count of lost calls if an arrival is rejected.
    - Schedules next arrival event: Sets the time for the next arrival in Method 1.
        """
        #var declaration
        freeServers = []
        #find out how many servers are free
        for i in range (len(self.servers)):
            if self.servers[i] == False:
                freeServers.append(i)
        
        #increment arrivals for M1
        self.totalArrivalsM1 += 1

        #if there are free servers considering the free spaces for handover calls
        if len(freeServers) > self.threshold:
            #freeServers[0] is the index of the first free server
            self.servers[freeServers[0]] = True
            self.timeNextEvent[freeServers[0]] = self.simTime + self.expService()
        #reject arrival
        else:
            self.totalLossM1 += 1

        #generate next arrival
        self.timeNextEvent[self.c] = self.simTime + self.expArrival(self.arrivalRateM1)

    #handover
    def arrivalM2(self):
        """
    Name: arrivalM2
    Description: Manages the arrival event for Method 2 in the simulation. It attempts to allocate a server
                 for the incoming call. If a server is available, the call is accepted and the server is marked
                 as busy. If no servers are available, the call is rejected. The function updates the simulation
                 state accordingly and schedules the next arrival event for Method 2.

    Inputs: None (Uses the object's current state and methods)
    Outputs: None (Modifies the object's state variables and schedules next events)

    - Updates self.totalArrivalsM2: Increments the count of total arrivals for Method 2.
    - Searches for a free server: Iterates through servers to find an available one.
    - Accepts or rejects arrivals: Based on the availability of a free server.
    - Updates self.servers: Marks a server as busy if an arrival is accepted.
    - Schedules next service event: Sets the time for the next service event if an arrival is accepted.
    - Updates self.totalLossM2: Increments the count of lost calls if an arrival is rejected.
    - Schedules next arrival event for Method 2: Sets the time for the next arrival in Method 2.
        """
        #vars declaration
        allocated = False
        #allocate arrival to free state var, give service time
        for i in range (len(self.servers)):
            if self.servers[i] == False:
                allocated = True
                break
            else:
                pass
        #alter state vars accordingly
        self.totalArrivalsM2 += 1
        if allocated == True:
            self.servers[i] = True
            self.timeNextEvent[i] = self.simTime + self.expService()
        else:
            self.totalLossM2 += 1
        #generate next arrival for handover
        self.timeNextEvent[self.c+1] = self.simTime + self.expArrival(self.arrivalRateM2)




    def depart(self, minIndex):
        """
    Name: depart
    Description: Simulates a departure event in the queuing system. This function is called when a call
                 or service completes and frees up a server. It updates the server status, resets the event
                 time for the departing server, and increments the total departure count.

    Inputs:
    - minIndex (int): The index of the server from which the departure is occurring.

    Outputs: None (Modifies the object's state variables)

    - Frees up the server at index minIndex: Sets the server status to False (idle).
    - Resets the time for the next event for the departing server: Sets it to -1, indicating no scheduled event.
    - Updates self.totalDepartures: Increments the count of total departures.
        """
        #free up server slot with index minIndex
        self.servers[minIndex] = False
        self.timeNextEvent[minIndex] = -1
        #set state vars
        self.totalDepartures += 1



    #timing function
    #find the next event 
    def timing(self):
        """
    Name: timing
    Description: Determines the next event in the queuing system simulation by finding the minimum 
                 non-negative value in the list of next event times. This function updates the system 
                 clock to the time of the next event and returns the index of that event.

    Inputs: None (Uses the object's current state)
    Outputs:
    - minIndex (int): The index of the next event in the 'self.timeNextEvent' list.

    - Finds the next event: Identifies the event with the minimum future time.
    - Updates self.simTime: Sets the simulation time to the time of the next event.
        """
        #find next event, min in next event times   
        mIndex = lambda lst: lst.index(min(x for x in lst if x >= 0))
        minIndex = mIndex(self.timeNextEvent)
        #update system clock
        self.simTime = self.timeNextEvent[minIndex]
        #update sim times inside 
        return minIndex


    def updateStats(self):
        """
    Name: updateStats
    Description: Updates the statistical data for server utilization in the queuing system simulation. 
                 This function calculates the amount of time each server has been active since the last 
                 update and updates the utilization statistics accordingly. It also updates the 'timeLast' 
                 variable to the current simulation time.

    Inputs: None (Uses the object's current state)
    Outputs: None (Modifies the object's state variables)

    - Updates self.areaServerStatus: For each active server, increments the server utilization time 
                                     since the last update.
    - Updates self.timeLast: Sets it to the current simulation time (self.simTime).
        """
        #for each server, if server is active increment server util based on simTime and timeLast
        for i in range(len(self.servers)):
            if self.servers[i] == True:
                self.areaServerStatus[i] += self.simTime - self.timeLast
        #update timeLast to be simTime
        self.timeLast = self.simTime


    #generates report
    def report(self):
        return {
            'c': self.c,
            'threshold': self.threshold,
            'arrivalRateM1': self.arrivalRateM1,
            'arrivalRateM2': self.arrivalRateM2,
            'serviceAverage': self.serviceAverage,
            'arrivalsForTerm': self.arrivalsForTerm,
            'servers': self.servers,
            'simTime': self.simTime,
            'timeLast': self.timeLast,
            'timeNextEvent': self.timeNextEvent,
            'totalArrivalsM1': self.totalArrivalsM1,
            'totalArrivalsM2': self.totalArrivalsM2,
            'totalDepartures': self.totalDepartures,
            'totalLossM1': self.totalLossM1,
            'totalLossM2': self.totalLossM2,
            'areaServerStatus': sum(self.areaServerStatus)/self.simTime
        }

    def runSimulation(self):
        """
    Name: runSimulation
    Description: Executes the queuing system simulation. It initializes the simulation, determines the next 
                 event (arrival or departure), updates statistics, and handles the events accordingly. The 
                 simulation runs until a specified termination condition is met, and then it generates a report.

    Inputs: None (Uses the object's initial state and parameters)
    Outputs:
    - Report of the simulation: Generated by the 'self.report()' method after simulation ends.

    - Initializes simulation: Sets the initial arrival times for both methods.
    - Main simulation loop: Continuously processes the next event until the termination condition is met.
    - Handles different types of events: Depending on the event type (arrivalM1, arrivalM2, or departure), 
                                         calls the appropriate function to handle the event.
    - Checks termination condition: The simulation stops when the combined total arrivals for both methods 
                                    reach the specified threshold ('self.arrivalsForTerm').
        """
        #self.resetStateVars()
        self.timeNextEvent[self.c] = self.expArrival(self.arrivalRateM1)
        self.timeNextEvent[self.c+1] = self.expArrival(self.arrivalRateM2)

        endSimVal = False
        while not endSimVal:
            #figure out next event using timing
            minIndex = self.timing()
            # Update statistics
            self.updateStats()
            #handle events
            #if handover call
            if minIndex == self.c + 1:
                self.arrivalM2()
            #if new call
            elif minIndex == self.c:
                self.arrivalM1()
            #if departure
            elif minIndex >= 0 and minIndex <= self.c-1:
                self.depart(minIndex)

            #termination end?
            if self.totalArrivalsM1 + self.totalArrivalsM2 >= self.arrivalsForTerm:
                endSimVal = True
            #function for it

        return self.report()





class AnalyticalModels:
    def __init__(self, c, threshold, arrivalM1, arrivalM2, serviceRate):
        self.c = c
        self.threshold = threshold
        self.arrivalM1 = arrivalM1
        self.arrivalM2 = arrivalM2
        self.serviceRate = serviceRate


    def initialSPs(self):
        #format values 
        trafficIntensity1 = (self.arrivalM1 + self.arrivalM2) / self.serviceRate
        trafficIntensity2 = (self.arrivalM2)/ self.serviceRate
        cMinusN = self.c - self.threshold
        LHS = sum((1/factorial(j))*(trafficIntensity1**j) for j in range(cMinusN+1))
        RHS = sum((1/factorial(j))*(trafficIntensity1**cMinusN)*(trafficIntensity2**(j-cMinusN)) for j in range(cMinusN+1, self.c+1))
        initialSP = 1/(LHS+RHS)
        return initialSP



    def stateProbabilities(self, k):
        #k is specified steady state
        #format vals
        trafficIntensity1 = (self.arrivalM1 + self.arrivalM2) / self.serviceRate
        trafficIntensity2 = (self.arrivalM2)/ self.serviceRate
        cMinusN = self.c - self.threshold
        if k <= cMinusN and k >= 0:
            SP = (1/factorial(k))*(trafficIntensity1**k)*self.initialSPs()
        elif k >= (cMinusN+1) and k <= self.c:
            SP = (1/factorial(k))*(trafficIntensity1**cMinusN)*(trafficIntensity2**(k-cMinusN))*self.initialSPs()
        else:
            SP = None
        return SP

    def newCallBP(self):
        cMinusN = self.c - self.threshold
        BP = sum(self.stateProbabilities(k) for k in range(cMinusN, self.c+1))
        return BP

    def handoverBP(self):
        return self.stateProbabilities(self.c)


    def ANoCiS(self):
        stateProbs = [self.stateProbabilities(k) for k in range(self.c+1)]
        averageCustomers = sum(k*stateProbs[k] for k in range(self.c+1))
        return averageCustomers


    def SU(self):
        averageCustomers = self.ANoCiS()
        averageSU = averageCustomers / self.c
        return averageSU


#------------------------------------------------validation methods, varying new call arrival rate------------------------------------------------
def SUgraphNc():
    arrivalRateNC = 0.01
    arrivalRateH = 0.1 # constant
    maxArrivalRateNC = 0.5
    arrivalRateNCStep = 0.005
    data = []
    serviceAverage = 100
    sampleSize = 10
    #for each arrival rate
    while arrivalRateNC <= maxArrivalRateNC:
        print(arrivalRateNC)
        #run sim 10 times for each arrival rate
        tempRow = [arrivalRateNC]
        for i in range(sampleSize):
            qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=arrivalRateNC, arrivalRateM2 = arrivalRateH, serviceAverage=serviceAverage, arrivalsForTerm=10000)
            repo = qSim.runSimulation()
            #acquire SU
            SUsample = repo["areaServerStatus"]/repo["c"]
            tempRow.append(SUsample)
        tempRow.append(sum(tempRow[1:])/sampleSize)
        #acquire analytical model for arrival rate
        aModel = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateNC, arrivalM2 = arrivalRateH, serviceRate = 1/serviceAverage)
        tBP = aModel.SU()
        tempRow.append(tBP)
        #store data
        data.append(tempRow)
        #increment arrival rate
        arrivalRateNC += arrivalRateNCStep
    #create graph
    # Convert data to a numpy array for easier slicing
    # Convert data to a numpy array for easier slicing
    dataArray = np.array(data)

    # x values are the first column in the data
    xValues = dataArray[:, 0]

    # y values are all columns except for the first (x values) and the last two (y average and y real)
    for ySeries in dataArray[:, 1:-2].T:  # .T to transpose and iterate over y values
        plt.plot(xValues, ySeries, color='grey', linewidth=0.5, alpha=0.5)  # faint lines

    # y average values are the second to last column
    yAverage = dataArray[:, -2]
    plt.plot(xValues, yAverage, label='Averaged', linewidth=2, color='blue')  # thick line for average

    # y real values are the last column
    yReal = dataArray[:, -1]
    plt.plot(xValues, yReal, label='Theoretical', linewidth=2, color='red')  # thick line for real

    # Adding labels and title
    plt.xlabel('new call arrival rate')
    plt.ylabel('Server utilisation')
    plt.title('New call arrival rate to Server utilisation')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

def BPgraphNc():
    arrivalRateNC = 0.01
    arrivalRateH = 0.03 # constant
    maxArrivalRateNC = 0.5
    arrivalRateNCStep = 0.005
    data = []
    serviceAverage = 100
    sampleSize = 10
    #for each arrival rate
    while arrivalRateNC <= maxArrivalRateNC:
        print(arrivalRateNC)
        #run sim 10 times for each arrival rate
        tempRow = [arrivalRateNC]
        for i in range(sampleSize):
            qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=arrivalRateNC, arrivalRateM2 = arrivalRateH, serviceAverage=serviceAverage, arrivalsForTerm=10000)
            repo = qSim.runSimulation()
            
            #acquire BPs, aggregate ------------------------------------
            BPH = repo["totalLossM2"]/repo["totalArrivalsM2"]
            BPnA = repo["totalLossM1"]/repo["totalArrivalsM1"]
            BPA = BPnA + (10*BPH)
            tempRow.append(BPA)
        tempRow.append(sum(tempRow[1:])/sampleSize)
        #acquire analytical model for arrival rate
        aModel = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateNC, arrivalM2 = arrivalRateH, serviceRate = 1/serviceAverage)
        
        #make aggregate for BP theoreticals---------------------------

        tBPnA = aModel.newCallBP()
        tBPH = aModel.handoverBP()
        tABP = tBPnA +(10*tBPH)

        tempRow.append(tABP)
        #store data
        data.append(tempRow)
        #increment arrival rate
        arrivalRateNC += arrivalRateNCStep
    #create graph
    # Convert data to a numpy array for easier slicing
    # Convert data to a numpy array for easier slicing
    dataArray = np.array(data)

    # x values are the first column in the data
    xValues = dataArray[:, 0]

    # y values are all columns except for the first (x values) and the last two (y average and y real)
    for ySeries in dataArray[:, 1:-2].T:  # .T to transpose and iterate over y values
        plt.plot(xValues, ySeries, color='grey', linewidth=0.5, alpha=0.5)  # faint lines

    # y average values are the second to last column
    yAverage = dataArray[:, -2]
    plt.plot(xValues, yAverage, label='Averaged', linewidth=2, color='blue')  # thick line for average

    # y real values are the last column
    yReal = dataArray[:, -1]
    plt.plot(xValues, yReal, label='Theoretical', linewidth=2, color='red')  # thick line for real

    # Adding labels and title
    plt.xlabel('new call arrival rate')
    plt.ylabel('Aggregated blocking probability')
    plt.title('New call arrival rate to aggregated blocking probability')
    plt.legend()
    plt.axhline(y=0.02, color='green', linestyle='--', label='Threshold (0.02)')
    # Find the intersection for the vertical line
    intersection_x = None
    for i in range(len(xValues)):
        if yReal[i] >= 0.02:
            intersection_x = xValues[i]
            break

    # Adding vertical line at the intersection point
    if intersection_x is not None:
        plt.axvline(x=intersection_x, color='orange', linestyle='--', label=f'Intersection at {intersection_x:.3f}')




    # Show the plot
    plt.grid(True)
    plt.show()

def NoCiSNc():
    arrivalRateNC = 0.01
    arrivalRateH = 0.1 # constant
    maxArrivalRateNC = 0.5
    arrivalRateNCStep = 0.005
    data = []
    serviceAverage = 100
    sampleSize = 10
    #for each arrival rate
    while arrivalRateNC <= maxArrivalRateNC:
        print(arrivalRateNC)
        #run sim 10 times for each arrival rate
        tempRow = [arrivalRateNC]
        for i in range(sampleSize):
            qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=arrivalRateNC, arrivalRateM2 = arrivalRateH, serviceAverage=serviceAverage, arrivalsForTerm=10000)
            repo = qSim.runSimulation()
            #acquire SU
            NOsample = repo["areaServerStatus"]
            tempRow.append(NOsample)
        tempRow.append(sum(tempRow[1:])/sampleSize)
        #acquire analytical model for arrival rate
        aModel = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateNC, arrivalM2 = arrivalRateH, serviceRate = 1/serviceAverage)
        tNO = aModel.ANoCiS()
        tempRow.append(tNO)
        #store data
        data.append(tempRow)
        #increment arrival rate
        arrivalRateNC += arrivalRateNCStep
    #create graph
    # Convert data to a numpy array for easier slicing
    # Convert data to a numpy array for easier slicing
    dataArray = np.array(data)

    # x values are the first column in the data
    xValues = dataArray[:, 0]

    # y values are all columns except for the first (x values) and the last two (y average and y real)
    for ySeries in dataArray[:, 1:-2].T:  # .T to transpose and iterate over y values
        plt.plot(xValues, ySeries, color='grey', linewidth=0.5, alpha=0.5)  # faint lines

    # y average values are the second to last column
    yAverage = dataArray[:, -2]
    plt.plot(xValues, yAverage, label='Averaged', linewidth=2, color='blue')  # thick line for average

    # y real values are the last column
    yReal = dataArray[:, -1]
    plt.plot(xValues, yReal, label='Theoretical', linewidth=2, color='red')  # thick line for real

    # Adding labels and title
    plt.xlabel('new call arrival rate')
    plt.ylabel('Average No. of customers in system')
    plt.title('New call arrival rate to Average No. of customers in system')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()



#------------------------------------------------validation methods, varying handover arrival rate------------------------------------------------
def SUgraphH():
    arrivalRateNC = 0.1 # constant
    arrivalRateH = 0.01 
    maxArrivalRateH = 0.5
    arrivalRateHStep = 0.005
    data = []
    serviceAverage = 100
    sampleSize = 10
    #for each arrival rate
    while arrivalRateH <= maxArrivalRateH:
        print(arrivalRateH)
        #run sim 10 times for each arrival rate
        tempRow = [arrivalRateH]
        for i in range(sampleSize):
            qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=arrivalRateNC, arrivalRateM2 = arrivalRateH, serviceAverage=serviceAverage, arrivalsForTerm=10000)
            repo = qSim.runSimulation()
            #acquire SU
            SUsample = repo["areaServerStatus"]/repo["c"]
            tempRow.append(SUsample)
        tempRow.append(sum(tempRow[1:])/sampleSize)
        #acquire analytical model for arrival rate
        aModel = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateNC, arrivalM2 = arrivalRateH, serviceRate = 1/serviceAverage)
        tBP = aModel.SU()
        tempRow.append(tBP)
        #store data
        data.append(tempRow)
        #increment arrival rate
        arrivalRateH += arrivalRateHStep
    #create graph
    # Convert data to a numpy array for easier slicing
    # Convert data to a numpy array for easier slicing
    dataArray = np.array(data)

    # x values are the first column in the data
    xValues = dataArray[:, 0]

    # y values are all columns except for the first (x values) and the last two (y average and y real)
    for ySeries in dataArray[:, 1:-2].T:  # .T to transpose and iterate over y values
        plt.plot(xValues, ySeries, color='grey', linewidth=0.5, alpha=0.5)  # faint lines

    # y average values are the second to last column
    yAverage = dataArray[:, -2]
    plt.plot(xValues, yAverage, label='Averaged', linewidth=2, color='blue')  # thick line for average

    # y real values are the last column
    yReal = dataArray[:, -1]
    plt.plot(xValues, yReal, label='Theoretical', linewidth=2, color='red')  # thick line for real

    # Adding labels and title
    plt.xlabel('Handover arrival rate')
    plt.ylabel('Server utilisation')
    plt.title('Handover arrival rate to Server utilisation')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

def BPgraphH():
    arrivalRateNC = 0.01
    arrivalRateH = 0.01 # constant
    maxArrivalRateH = 0.5
    arrivalRateHStep = 0.005
    data = []
    serviceAverage = 100
    sampleSize = 10
    #for each arrival rate
    while arrivalRateH <= maxArrivalRateH:
        print(arrivalRateH)
        #run sim 10 times for each arrival rate
        tempRow = [arrivalRateH]
        for i in range(sampleSize):
            qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=arrivalRateNC, arrivalRateM2 = arrivalRateH, serviceAverage=serviceAverage, arrivalsForTerm=10000)
            repo = qSim.runSimulation()
            
            #acquire BPs, aggregate ------------------------------------
            BPH = repo["totalLossM2"]/repo["totalArrivalsM2"]
            BPnA = repo["totalLossM1"]/repo["totalArrivalsM1"]
            BPA = BPnA + (10*BPH)
            tempRow.append(BPA)
        tempRow.append(sum(tempRow[1:])/sampleSize)
        #acquire analytical model for arrival rate
        aModel = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateNC, arrivalM2 = arrivalRateH, serviceRate = 1/serviceAverage)
        
        #make aggregate for BP theoreticals---------------------------

        tBPnA = aModel.newCallBP()
        tBPH = aModel.handoverBP()
        tABP = tBPnA +(10*tBPH)

        tempRow.append(tABP)
        #store data
        data.append(tempRow)
        #increment arrival rate
        arrivalRateH += arrivalRateHStep
    #create graph
    # Convert data to a numpy array for easier slicing
    # Convert data to a numpy array for easier slicing
    dataArray = np.array(data)

    # x values are the first column in the data
    xValues = dataArray[:, 0]

    # y values are all columns except for the first (x values) and the last two (y average and y real)
    for ySeries in dataArray[:, 1:-2].T:  # .T to transpose and iterate over y values
        plt.plot(xValues, ySeries, color='grey', linewidth=0.5, alpha=0.5)  # faint lines

    # y average values are the second to last column
    yAverage = dataArray[:, -2]
    plt.plot(xValues, yAverage, label='Averaged', linewidth=2, color='blue')  # thick line for average

    # y real values are the last column
    yReal = dataArray[:, -1]
    plt.plot(xValues, yReal, label='Theoretical', linewidth=2, color='red')  # thick line for real

    # Adding labels and title
    plt.xlabel('Handover arrival rate')
    plt.ylabel('Aggregated blocking probability')
    plt.title('Handover arrival rate to aggregated blocking probability')
    plt.legend()

    # Show the plot


    plt.grid(True)
    plt.show()

def NoCiSH():

    arrivalRateNC = 0.1  # constant
    arrivalRateH = 0.01
    maxArrivalRateH = 0.5
    arrivalRateHStep = 0.005
    data = []
    serviceAverage = 100
    sampleSize = 10
    #for each arrival rate
    while arrivalRateH <= maxArrivalRateH:
        print(arrivalRateH)
        #run sim 10 times for each arrival rate
        tempRow = [arrivalRateH]
        for i in range(sampleSize):
            qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=arrivalRateNC, arrivalRateM2 = arrivalRateH, serviceAverage=serviceAverage, arrivalsForTerm=10000)
            repo = qSim.runSimulation()
            #acquire SU
            NOsample = repo["areaServerStatus"]
            tempRow.append(NOsample)
        tempRow.append(sum(tempRow[1:])/sampleSize)
        #acquire analytical model for arrival rate
        aModel = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateNC, arrivalM2 = arrivalRateH, serviceRate = 1/serviceAverage)
        tNO = aModel.ANoCiS()
        tempRow.append(tNO)
        #store data
        data.append(tempRow)
        #increment arrival rate
        arrivalRateH += arrivalRateHStep
    #create graph
    # Convert data to a numpy array for easier slicing
    # Convert data to a numpy array for easier slicing
    dataArray = np.array(data)

    # x values are the first column in the data
    xValues = dataArray[:, 0]

    # y values are all columns except for the first (x values) and the last two (y average and y real)
    for ySeries in dataArray[:, 1:-2].T:  # .T to transpose and iterate over y values
        plt.plot(xValues, ySeries, color='grey', linewidth=0.5, alpha=0.5)  # faint lines

    # y average values are the second to last column
    yAverage = dataArray[:, -2]
    plt.plot(xValues, yAverage, label='Averaged', linewidth=2, color='blue')  # thick line for average

    # y real values are the last column
    yReal = dataArray[:, -1]
    plt.plot(xValues, yReal, label='Theoretical', linewidth=2, color='red')  # thick line for real

    # Adding labels and title
    plt.xlabel('Handover arrival rate')
    plt.ylabel('Average No. of customers in system')
    plt.title('Handover arrival rate to Average No. of customers in system')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def RMSQ():
    arrivalRateNC = 0.01  # constant
    arrivalRateH = 0.1
    maxArrivalRateNC = 0.5
    arrivalRateNCStep = 0.005
    dataSU = []
    dataBP = []
    dataAN = []
    serviceAverage = 100
    sampleSize = 10
    #for each arrival rate
    while arrivalRateNC <= maxArrivalRateNC:
        print(arrivalRateNC)
        tempRowBP = []
        tempRowSU = []
        tempRowAN = []
        for i in range(sampleSize):
            qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=arrivalRateNC, arrivalRateM2 = arrivalRateH, serviceAverage=serviceAverage, arrivalsForTerm=10000)
            repo = qSim.runSimulation()
            #acquire SU
            SUsample = repo["areaServerStatus"]/repo["c"]
            #acquire BP
            BPH = repo["totalLossM2"]/repo["totalArrivalsM2"]
            BPnA = repo["totalLossM1"]/repo["totalArrivalsM1"]
            BPsample = BPnA + (10*BPH)
            #acquire AN
            ANsample = repo["areaServerStatus"]
            #add to corresponding data structures
            tempRowBP.append(BPsample)
            tempRowSU.append(SUsample)
            tempRowAN.append(ANsample)
        #average Each, add to overarching dataset
        BPaver = sum(tempRowBP)/sampleSize
        SUaver = sum(tempRowSU)/sampleSize
        ANaver = sum(tempRowAN)/sampleSize
        #comp to theoretical, add to dataset
        aModel = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateNC, arrivalM2 = arrivalRateH, serviceRate = 1/serviceAverage)
        tBPnA = aModel.newCallBP()
        tBPH = aModel.handoverBP()
        tAB = tBPnA +(10*tBPH)
        tSU = aModel.SU()
        tAN = aModel.ANoCiS()
        #real, theorietical
        dataBP.append([BPaver, tAB])
        dataSU.append([SUaver,tSU])
        dataAN.append([ANaver, tAN])

        #increment arrivalRate
        arrivalRateNC += arrivalRateNCStep
    rms = lambda data: sqrt(sum((x1-x2)**2 for x1, x2 in data) / len(data))
    print("BP, SU, AN")
    print(rms(dataBP))
    print(rms(dataSU))
    print(rms(dataAN))


def findHandover():
    # Parameters
    c = 16
    threshold = 2
    serviceAverage = 100
    arrivalsForTerm = 10000  # Adjust as needed
    arrivalRateM2 = 0.03

    # Binary search bounds
    low_bound = 0.0
    high_bound = 0.1  # Adjust based on expected maximum
    precision = 0.000001  # Define the precision of your search

    # Binary search loop
    while high_bound - low_bound > precision:
        print("doing")
        arrivalRateM1 = (low_bound + high_bound) / 2
        #simulation = QueueSimulation(c, threshold, arrivalRateM1, arrivalRateM2, serviceAverage, arrivalsForTerm)
        #report = simulation.runSimulation()
        # Calculate probabilities
        #CBP = report['totalLossM1'] / report['totalArrivalsM1']
        #HFP = report['totalLossM2'] / report['totalArrivalsM2']
        #ABP = CBP + 10 * HFP

        analytical = AnalyticalModels(c=16, threshold=2, arrivalM1 = arrivalRateM1, arrivalM2 = arrivalRateM2, serviceRate = 1/serviceAverage)
        ABP = (analytical.newCallBP())+10*(analytical.handoverBP())
        print(ABP)
        # Adjust bounds based on ABP
        if ABP < 0.02:
            low_bound = arrivalRateM1
        else:
            high_bound = arrivalRateM1
        print(f"hb = {high_bound}\nlb = {low_bound}\ndiff = {high_bound-low_bound}")
    # Result
    max_handover_rate = low_bound
    print(f"The maximum handover rate with ABP < 0.02 is approximately: {max_handover_rate}")



qSim = QueueSimulation(c=16, threshold=2, arrivalRateM1=0.01, arrivalRateM2 = 0.1, serviceAverage=1/100, arrivalsForTerm=10000)
print(qSim.runSimulation())
