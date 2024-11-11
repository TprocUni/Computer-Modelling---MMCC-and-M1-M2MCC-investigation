#Graph creation
import pickle
import matplotlib.pyplot as plt
from math import sqrt

def blockingPlot():
    # Open the pickle 
    with open('pickles\\blockingResults2.pkl', 'rb') as file:
        data = pickle.load(file)

    # Now, your_2d_list contains the data from the pickle file
    print(data)


    #converty to list of points
    noOfLines = len(data[0])
    #produces n lines, 1st line is theoretical, last is averaged
    linesList = [[] for _ in range(noOfLines)]
    for i in data:
        for line in range(1, noOfLines):
            linesList[line-1].append([i[0], i[line]])
        linesList[noOfLines-1].append([i[0], sum(i[2:])/(noOfLines-2)])
    temp = linesList[1]
    linesList[1] = linesList[noOfLines-1]
    linesList[noOfLines-1] = temp

    #plot graph
    for i, line in enumerate(linesList):
        x_values, y_values = zip(*line)  # Unzip the points into separate x and y arrays
        if i in [i for i in range(2,noOfLines)]:
            plt.plot(x_values,y_values, alpha = 0.25, linewidth = 0.5)
        else:
            plt.plot(x_values, y_values, linewidth = 1.5) 

    plt.axvline(x=0.1, color='red', linestyle='--')

    # Label the axes and add a legend
    plt.xlabel('Arrival rate')
    plt.ylabel('Blocking probabilty')
    plt.title("Comparing arrival rate to blocking probability, 10 runs, 500000 arrival termination")
    plt.legend(['Theoretical BP', "Averaged simulation BP",])

    # Show the plot
    plt.grid(True)
    plt.show()



def utilPlot():
    # Open the pickle 
    with open('pickles\\utilResults5.pkl', 'rb') as file:
        data = pickle.load(file)

    # Now, your_2d_list contains the data from the pickle file
    print(data)


    #converty to list of points
    noOfLines = len(data[0])
    #produces n lines, 1st line is theoretical, last is averaged
    linesList = [[] for _ in range(noOfLines)]
    for i in data:
        for line in range(1, noOfLines):
            linesList[line-1].append([i[0], i[line]])
        linesList[noOfLines-1].append([i[0], sum(i[2:])/(noOfLines-2)])
    temp = linesList[1]
    linesList[1] = linesList[noOfLines-1]
    linesList[noOfLines-1] = temp

    #plot graph
    for i, line in enumerate(linesList):
        x_values, y_values = zip(*line)  # Unzip the points into separate x and y arrays
        if i in [i for i in range(2,noOfLines)]:
            plt.plot(x_values,y_values, alpha = 0.3, linewidth = 0.5)
        else:
            plt.plot(x_values, y_values, linewidth = 1.5) 

    plt.axvline(x=0.1, color='red', linestyle='--')

    # Label the axes and add a legend
    plt.xlabel('Arrival rate')
    plt.ylabel('Server Utilisation')
    plt.title("Comparing arrival rate to server utilisation, 10 runs, 5000 arrival termination, inc. BP and warm-up")
    plt.legend(['Theoretical utilisation', "Averaged simulation utilisation",])

    # Show the plot
    plt.grid(True)
    plt.show()




def custPlot():
    # Open the pickle 
    with open('pickles\\NoOfCustResults2.pkl', 'rb') as file:
        data = pickle.load(file)

    # Now, your_2d_list contains the data from the pickle file
    print(data)


    #converty to list of points
    noOfLines = len(data[0])
    #produces n lines, 1st line is theoretical, last is averaged
    linesList = [[] for _ in range(noOfLines)]
    for i in data:
        for line in range(1, noOfLines):
            linesList[line-1].append([i[0], i[line]])
        linesList[noOfLines-1].append([i[0], sum(i[2:])/(noOfLines-2)])
    temp = linesList[1]
    linesList[1] = linesList[noOfLines-1]
    linesList[noOfLines-1] = temp

    #plot graph
    for i, line in enumerate(linesList):
        x_values, y_values = zip(*line)  # Unzip the points into separate x and y arrays
        if i in [i for i in range(2,noOfLines)]:
            plt.plot(x_values,y_values, alpha = 0.3, linewidth = 0.5)
        else:
            plt.plot(x_values, y_values, linewidth = 1.5) 

    plt.axvline(x=0.1, color='red', linestyle='--')

    # Label the axes and add a legend
    plt.xlabel('Arrival rate')
    plt.ylabel('No. Of customers in system')
    plt.title("Comparing arrival rate to No of customers in system, 10 runs, 500000 arrival termination")
    plt.legend(['Theoretical customers in system', "Averaged No. of customers in simulation",])

    # Show the plot
    plt.grid(True)
    plt.show()


def rootMSq():
    #data structures
    theoreticals = []
    averages = []
    # Open the pickle 
    with open('pickles\\blockingResults1.pkl', 'rb') as file:
        data = pickle.load(file)

    # Now, your_2d_list contains the data from the pickle file
    print(data)


    #converty to list of points
    noOfLines = len(data[0])
    for i in data:
        #get theoretical
        theoreticals.append(i[1])
        #get average
        averages.append(sum(i[2:])/(noOfLines-2))
    
    #calc root mean squared
    # Ensure both lists have the same length
    if len(theoreticals) != len(averages):
        raise ValueError("The lists must have the same length.")

    # Calculate the squared differences and their mean
    squaredDiffs = [(theor - avg) ** 2 for theor, avg in zip(theoreticals, averages)]
    meanSquaredDiff = sum(squaredDiffs) / len(squaredDiffs)
    #save 
    rmse = sqrt(meanSquaredDiff)
    print("The RMSE is:", rmse)



#custPlot()


rootMSq()
