
import numpy as np
from math import sqrt, ceil, floor
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info.states.densitymatrix import DensityMatrix

class CoincidenceMonitor:
    
    #Manages the CoincidenceMonitor, time_interval is the length of the intervals
    # in seconds, coincidence window is the difference in the time allowed to 
    # determine entanglement.
    def __init__(self, reported_alice=[], reported_bob=[]) -> None:
        self.reported_alice = np.array(reported_alice)
        self.reported_bob = np.array(reported_bob)
        self.coincidence_list = []

    # Finds the average rate, given the interval length, last element, and total elements
    def __find_rate(self, interval_len, last_ele, num_ele):
        
        #Max time value
        time = ceil(last_ele[0])
        num_intervals = time / interval_len

        #Average count per time interval
        return num_ele / num_intervals


    #Calculate the number of clicks that match the dark count flag with the given flag. 
    # If the dark count flag is 1, this measures the number of false-positive coincidences.
    # If the dark count flag is 0, measures the number of true positive coincidences.
    def count_matches( self, flag, parsed_list):
        ele_count = 0
        for el in parsed_list:
            if(el[2] == flag):
                ele_count+=1
        return ele_count

    #Calculate the rate of clicks that match the dark count flag with the given flag. 
    # If the dark count flag is 1, this measures the number of false-positive coincidences.
    # If the dark count flag is 0, measures the number of true positive coincidences.
    def calculate_rate( self, flag = 1, interval_len = 1, parsed_list=[]):
        ele_count = self.count_matches(flag, parsed_list)
        
        #Last element in the sorted parsed_list of reported values
        last_ele = parsed_list[-1]

        return self.__find_rate(interval_len, last_ele, ele_count)
    
    def calculate_percent( self, flag = 1, parsed_list=[]):
        
        ele_count = self.count_matches(flag, parsed_list)

        return ele_count / len(parsed_list)

    # Calculate average count rate within specific time intervals
    # If the interval is zero or less return the total number 
    # of counts. Default is total count.
    def count_rate( self, parsed_list, interval_len = 1):

        #Return number of rows, if the interval is less than zero
        if( interval_len <= 0 ):
            return parsed_list.shape[0]

        return ({
            'true': (self.calculate_rate(flag=0, interval_len=1, parsed_list=parsed_list), self.calculate_percent(flag=0, parsed_list=parsed_list)),
            'false': (self.calculate_rate(flag=1, interval_len=1, parsed_list=parsed_list), self.calculate_percent(flag=1, parsed_list=parsed_list))
        })
        
    
    # Counts the number of coincidences that match the flag in the third element (dark count flag).
    # If the dark count flag is 1, this measures the number of false-positive coincidences.
    # If the dark count flag is 0, measures the number of true positive coincidences.
    def coincidence_count_elements( self, flag):
        match_count = 0

        #Iterate over list
        for i in range(self.coincidence_list.shape[0]):
            #When counting false-positives, check if either alice or bob's measurement was a dark count.
            if flag == 1 and (self.coincidence_list[i][0][2] == flag or self.coincidence_list[i][1][2] == flag):
                match_count += 1
            #When counting true positives, check if alice and bob's measurements were not dark counts
            elif flag == 0 and (self.coincidence_list[i][0][2] == flag and self.coincidence_list[i][1][2] == flag):
                match_count += 1
        return match_count

    #Calculate the rate of coincidences that match the dark count flag with the given flag. 
    # If the dark count flag is 1, this measures the number of false-positive coincidences.
    # If the dark count flag is 0, measures the number of true positive coincidences.
    def coincidence_rate_helper( self, flag = 1, interval_len = 1):
        
        #Find the coincidences where their dark count flag matches the given flag
        match_count = self.coincidence_count_elements(flag)
        
        #Last element in the sorted list of reported values
        last_ele = self.coincidence_list[self.coincidence_list.shape[0]-1][0]

        return self.__find_rate(interval_len, last_ele, match_count)
    
    def coincidence_percent_helper( self, flag = 1):
        
        false_pos_count = self.coincidence_count_elements(flag)

        return false_pos_count / self.coincidence_list.shape[0]

    def __b_search_helper(self, low_index, high_index, window, alice_index):
    
        #Base condition, if there is no more array left to search: stop
        if( high_index < low_index ):
            return
        
        mid_index = (high_index + low_index) // 2

        # The difference between alice and bob's timestamps at the indexes of interest
        reported_diff = self.reported_alice[alice_index][0] - self.reported_bob[mid_index][0]

        #If the absolute value of the difference is within the given coincidence window, add this to the coincidence list
        if( abs(reported_diff) <= window ):
            self.coincidence_list.append([self.reported_alice[alice_index], self.reported_bob[mid_index]])
            return
        
        #If the reported difference is negative, use the left half of bob's array; otherwise, use the right
        elif( reported_diff < 0 ):
            self.__b_search_helper(low_index, mid_index - 1, window, alice_index)
            return
        else:
            self.__b_search_helper(mid_index + 1, high_index, window, alice_index)
            return

    # Generates the coincidence list based on the window given
    def __generate_coincidence_list(self, window):
        
        #For every measurement alice took, attempt to find a coincidence in bob's list
        for i in range(self.reported_alice.shape[0]):
            self.__b_search_helper(0, self.reported_bob.shape[0] - 1, window, i)
    
    #Count the rate of entanglement by finding the coincidences and averaging the 
    # rate over the given time interval
    def coincidence_rate(self, interval = 1, window = 0.000000001):

        #Generate the coincidence list
        self.__generate_coincidence_list(window)

        #Ensure the coincidence list is a numpy array
        self.coincidence_list = np.array(self.coincidence_list)
        
        #Return zero if there are no coincidences
        if len(self.coincidence_list) == 0:
            return ({
                'true': (0, 0),
                'false': (0, 0)
            })
        
        #Return number of rows, if the interval is less than zero
        if( interval <= 0 ):
            return self.coincidence_list.shape[0]
        
        #Return the false and positive coincidences
        return ({
            'true': (self.coincidence_rate_helper(flag=0, interval_len=1), self.coincidence_percent_helper(flag=0)),
            'false': (self.coincidence_rate_helper(flag=1, interval_len=1), self.coincidence_percent_helper(flag=1))
        })
    
    #Calcualte fidelity of a state based on the measurement values and original density matrix
    def fidelity( self, measurement_vals, original_density_matrix):
        
        #convert to Qiskit DensityMatrix
        desired_dm = DensityMatrix(np.matrix(original_density_matrix))

        total_measurements = measurement_vals['00'] + measurement_vals['01'] + measurement_vals['10'] + measurement_vals['11']

        #Obtain experimental probability amplitudes of each possible measurement
        p_00 = sqrt(measurement_vals['00'] / total_measurements)
        p_01 = sqrt(measurement_vals['01'] / total_measurements)
        p_10 = sqrt(measurement_vals['10'] / total_measurements)
        p_11 = sqrt(measurement_vals['11'] / total_measurements)
        
        #Column vector of the state
        state_vector = p_00*np.kron([1,0], [1,0]) + p_01*np.kron([1,0],[0,1]) + p_10*np.kron([0,1], [1,0]) + p_11*np.kron([0,1],[0,1])
        
        
        #calculate the experimentally formulated density matrix
        # Convert this to a Qiskit DensityMatrix
        measured_den = DensityMatrix(np.dot(np.reshape(state_vector, (-1,1)), np.reshape(state_vector, (1,4))))

        return state_fidelity(desired_dm, measured_den)
    
    
    


