from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
from error_post import PostProcessor
import multiprocessing
import time
from pathlib import Path
import sys

class QiskitSimulation:
    '''
    Main class to create the circuit and run simulations
    '''
    def __init__(self, density_matrix=[[]], case=1) -> None:
        # Parse and store density matrix
        self.density_matrix = np.matrix(density_matrix)
        # Parse and store the case in question
        self.case = str(case)
        # Parse and store the mode. 1 is for complete simulation, 2 is
        # pre-computed measurements.
        self.mode = 1
        if len(sys.argv) > 1:
            try:
                self.mode = int(sys.argv[1])
                if self.mode != 1 and self.mode != 2:
                    print('Mode has to be 1 or 2')
                    sys.exit()
            except Exception as e:
                print('Mode has to be an integer')
                sys.exit()
        # Run simulation
        self.run_simulation()
    
    def run_bell_inequality_test(self, zz, zx, xz, xx):
        '''
        Compute Bell inequality value
        '''
        # Parse results fo ZZ bases
        chshzz1 = 0
        chshzz2 = 0
        no_shotszz = 0
        for element in zz:
            parity = (-1)**(int(element[0])+int(element[1]))
            chshzz1 += parity*zz[element]
            chshzz2 += parity*zz[element]
            no_shotszz += zz[element]
        # Parse results fo ZX bases
        chshzx1 = 0
        chshzx2 = 0
        no_shotszx = 0
        for element in zx:
            parity = (-1)**(int(element[0])+int(element[1]))
            chshzx1 += parity*zx[element]
            chshzx2 -= parity*zx[element]
            no_shotszx += zx[element]
        # Parse results fo XZ bases
        chshxz1 = 0
        chshxz2 = 0
        no_shotsxz = 0
        for element in xz:
            parity = (-1)**(int(element[0])+int(element[1]))
            chshxz1 -= parity*xz[element]
            chshxz2 += parity*xz[element]
            no_shotsxz += xz[element]
        # Parse results fo XX bases
        chshxx1 = 0
        chshxx2 = 0
        no_shotsxx = 0
        for element in xx:
            parity = (-1)**(int(element[0])+int(element[1]))
            chshxx1 += parity*xx[element]
            chshxx2 += parity*xx[element]
            no_shotsxx += xx[element]
        # Comput CHSH values. Subtractions are handled under each loop, so these are all additions
        chsh1 = (chshzz1/no_shotszz) + (chshzx1/no_shotszx) + (chshxz1/no_shotsxz) + (chshxx1/no_shotsxx)
        chsh2 = (chshzz2/no_shotszz) + (chshzx2/no_shotszx) + (chshxz2/no_shotsxz) + (chshxx2/no_shotsxx)
        return (chsh1, chsh2)
    
    def find_measurements(self, sim, qc):
        '''
        Run simulation, find measurement for 1 shot, and return the results.
        Is parallelized using multiprocessing
        '''
        result = sim.run(qc, shots=1).result()
        values = list(result.get_counts().keys())[0]
        return (values[0], values[1], result.time_taken)

    def run_simulation(self):
        '''
        Run the simulation
        '''
        sim = AerSimulator()
        SECONDS = 30
        # Alice measurement base between 0 and 2*pi
        number_of_thetas = 15
        theta_vec = np.linspace(0, 2*np.pi, number_of_thetas)
        theta_counter = 1
        for theta in theta_vec:
            # Bob's bases Z and X. Alice's bases Z + theta, X + theta. All 4 combinations considered
            obs_vec = ['ZZ', 'ZX', 'XZ', 'XX']
            xx = []
            xz = []
            zx = []
            zz = []
            for obs in obs_vec:
                alice = []
                bob = []
                if self.mode == 1:
                    # Complete simulation case
                    # Create circuit
                    qc = QuantumCircuit(2,2)
                    qc.set_density_matrix(self.density_matrix)
                    # Rotate in bloch sphere
                    qc.ry(theta, 0)
                    for a in range(2):
                        if obs[a] == 'X':
                            qc.h(a)
                    qc.measure(range(2),range(2))
                    start_time = time.time()
                    print('\nGenerating measurements for case', obs)
                    results = []
                    total = 0
                    for i in range(SECONDS):
                        with multiprocessing.Pool() as pool:
                            async_results = []
                            for j in range(int(np.random.poisson(15000, 1))):
                                # Parallelize each shot computation for faster results
                                async_results.append(pool.apply_async(self.find_measurements, args=(sim, qc)))
                            for ar in async_results:
                                # Get results from async execution
                                res = ar.get()
                                results.append(res)
                                total += res[2]
                        pool.close()
                    timer = 0
                    for a_m, b_m, t in results:
                        timer += (t / total)
                        alice.append([timer * SECONDS, int(a_m), 0])
                        bob.append([timer * SECONDS, int(b_m), 0])
                    # Store results in CSV, commented by default. If want to store measurements, uncomment this
                    # try:
                    #     parties = ['Alice', 'Bob']
                    #     for party in parties:
                    #         path = './data/' + self.case + '/' + str(theta_counter) + '/' + str(obs) +  '/' + str(SECONDS)
                    #         if not Path(path).exists():
                    #             Path(path).mkdir(parents=True, exist_ok=True)
                    #         with open(path + '/' + party + '.csv', 'w') as file:
                    #             res = alice
                    #             if party == 'Bob':
                    #                 res = bob
                    #             for line in res:
                    #                 file.write('{},{},{}'.format(str(line[0]), str(line[1]), str(line[2])))
                    #                 file.write('\n')
                    #         file.close()
                    # except Exception as e:
                    #     print('Some error occurred', e)
                    #     sys.exit()
                    end_time = time.time()
                    print(end_time - start_time, 'seconds for measurements')
                elif self.mode == 2:
                    # Use pre-computed measurements
                    try:
                        alice_file = open('./data/' + self.case + '/' + str(theta_counter) + '/' + str(obs) +  '/' + str(SECONDS) + '/Alice.csv')
                        bob_file = open('./data/' + self.case + '/' + str(theta_counter) + '/' + str(obs) +  '/' + str(SECONDS) + '/Bob.csv')
                        alice = np.genfromtxt(alice_file, delimiter=',')
                        bob = np.genfromtxt(bob_file, delimiter=',')
                    except Exception as e:
                        print('Some error occurred', e)
                        sys.exit()
                print('\nMeasurements generated, running post processing')
                start_time = time.time()
                # Run post processing
                post_processor = PostProcessor(alice, bob, self.density_matrix)
                if obs == 'ZZ':
                    zz = post_processor.total_coincidence_rates
                elif obs == 'ZX':
                    zx = post_processor.total_coincidence_rates
                elif obs == 'XZ':
                    xz = post_processor.total_coincidence_rates
                else:
                    xx = post_processor.total_coincidence_rates
                print('\nPost processing completed')
                print('\nTotal count rate:', post_processor.count_rates)
                print('\nCoincidence rate:', post_processor.coincidence_rates)
                print('\nFidelity:', post_processor.fidelity)
                end_time = time.time()
                print('\n', end_time - start_time, 'seconds for postprocessing')
            print('\nCHSH inequality output: ', self.run_bell_inequality_test(zz, zx, xz, xx))
            theta_counter += 1

# Case: Maximally entangled EPR pairs
print('\nRunning analysis for case 1: Maximally entangled EPR pairs\n')
QiskitSimulation(density_matrix=[[1/2,0,0,1/2], [0,0,0,0], [0,0,0,0], [1/2,0,0,1/2]], case=1)
# Case: Maximally entangled EPR pairs with mixture
print('\nRunning analysis for case 2: Maximally entangled EPR pairs with mixture\n')
QiskitSimulation(density_matrix=[[7/20,0,0,7/20], [0,3/20,0,0], [0,0,3/20,0], [7/20,0,0,7/20]], case=2)
# Case: Non-maximally entangled pairs
print('\nRunning analysis for case 3: Non-maximally entangled pairs\n')
QiskitSimulation(density_matrix=[[1/5,0,0,2/5], [0,0,0,0], [0,0,0,0], [2/5,0,0,4/5]], case=3)
# Case: Non-maximally entangled pairs with mixture
print('\nRunning analysis for case 4: Non-maximally entangled pairs with mixture\n')
QiskitSimulation(density_matrix=[[7/50,0,0,7/25], [0,3/20,0,0], [0,0,3/20,0], [7/25,0,0,14/25]], case=4)