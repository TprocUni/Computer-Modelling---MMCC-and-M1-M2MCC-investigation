from MMCC_queue import arrive, depart, timing, resetStateVars, simTime, servers, timeNextEvent, totalArrivals, totalDepartures

def test_arrival():
    resetStateVars()
    print("Testing Arrival Function")
    # Initially, no servers should be busy
    assert not any(servers), "Error: Some servers are initially busy"

    # Simulate an arrival
    arrive()
    assert totalArrivals == 1, "Error: Total arrivals not updated correctly"
    assert any(servers), "Error: No server is busy after arrival"
    print("Arrival Function Passed")

def test_departure():
    resetStateVars()
    print("Testing Departure Function")
    # Simulate an arrival and then a departure
    arrive()
    minIndex = servers.index(True)
    depart(minIndex)
    assert totalDepartures == 1, "Error: Total departures not updated correctly"
    assert not servers[minIndex], "Error: Server not freed after departure"
    print("Departure Function Passed")

def test_timing():
    resetStateVars()
    print("Testing Timing Function")
    # Simulate an event and check timing
    original_time = simTime
    arrive()
    minIndex = timing()
    assert simTime > original_time, "Error: Simulation time not updated correctly"
    assert minIndex == servers.index(True), "Error: Incorrect next event index"
    print("Timing Function Passed")

if __name__ == "__main__":
    test_arrival()
    test_departure()
    test_timing()
