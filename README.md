The project focuses on simulating and analysing two queueing models: the M/M/C/C system and the M1+M2/M/C/C system, which are useful in understanding the behaviour of cellular networks, healthcare and transport systems, air traffic control, amongst many others.

## M/M/C/C Queueing System
Description: Represents a system with multiple servers (C), where both arrival and service times are exponentially distributed (Markovian). There is no waiting space; if all servers are occupied, incoming calls are blocked.

Implementation: Utilises a procedural programming approach. Key components include event handling for arrivals and departures, state management of servers, and timing functions to progress the simulation.

Performance Metrics:

Blocking Probability: The likelihood that an incoming call is blocked due to all servers being busy.

Server Utilisation: The proportion of time servers are actively engaged in service.
Average Number of Customers: The mean number of customers in the system over time.

Validation: Results from the simulation are compared against analytical models using metrics like the root mean squared error to ensure accuracy.

## M1+M2/M/C/C Queueing System
Description: An extension of the M/M/C/C model introducing two types of calls:

New Calls (M1): Regular incoming calls.

Handover Calls (M2): Calls transferred from adjacent cells, given higher priority.

Implementation: Employs object-oriented programming to handle the increased complexity. The model includes a priority mechanism to reflect real-world cellular networks where handover calls are less likely to be dropped.

Performance Metrics:

Aggregated Blocking Probability: Combines the blocking probabilities of new and handover calls, with handover calls weighted more heavily due to their priority.

Server Utilisation and Customer Count: Similar metrics as the M/M/C/C model but adjusted for the two types of calls.

Validation: Ensures the simulation aligns with theoretical expectations across various arrival rates for both new and handover calls.

## Key Project Components

Design and Code Overview: Detailed planning of variables, state management, event handling, and the main simulation flow for both models.

Statistical Measures: Calculation of key performance indicators to assess system efficiency and capacity.

Testing and Validation: Rigorous testing of simulation functions and validation against analytical formulas to confirm model accuracy.

Exercises and Solutions: Addressed specific scenarios, such as determining maximum input rates while keeping blocking probabilities below a certain threshold.

Warm-up Period Analysis: Investigated the initial phase of the simulation to ensure the system reaches a steady state before data collection.

Improvements and Further Research: Suggested enhancements like dynamic server selection algorithms and predictive maintenance strategies to improve simulation realism and system performance.
