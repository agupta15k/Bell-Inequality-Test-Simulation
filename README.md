# Project - Bell Inequality Test simulation

## Problem statement

Write a program in Python to simulate Bell Inequality Test. Control the entanglement source to generate the following 4 states:

* Maximally entangled EPR pairs: |Φ+⟩ = 1/√2(|HH⟩ + |VV⟩)
* Maximally entangled EPR pairs with mixture: ρ = 0.7|Φ+⟩⟨Φ+| + 0.15|HV⟩⟨HV| + 0.15|VH⟩⟨VH|
* Non-maximally entangled pairs: |ϕ⟩ = 1/√5|HH⟩ + √2/5|VV⟩
* Non-maximally entangled pairs with mixture: ρ′ = 0.7|ϕ⟩⟨ϕ| + 0.15|HV⟩⟨HV| + 0.15|VH⟩⟨VH|

For each of the four states, calculate total count rate, coincidence rate, fidelity, and Bell inequality (BI).

## Specifications

* Entangled pairs are generated from a Poisson statistics with average rate of 15,000 entanglements per second.
* Detector has 10% efficiency, 1,000 Hz dark count rate, and 4 microseconds dead time.
* Both parties can measure in arbitrary bases HH, VV, DA, etc.
* Coincidence window is 1 nanosecond.
* Simulation should run for 30 seconds.

## Process

**Some considerations**
* Our systems not being able to properly simulate 1 nanosecond coincidence window coupled with the fact that Qiskit simulations take more than 1 millisecond on average, did not allow us to directly simulate the exact setup.
* For simulation, we performed the measurements using Qiskit, but scaled down the timestamps to be within 30 seconds. Since the timestamps were scaled down, we could simulate 1 nanosecond coincidence window and 4 microsecods dead time.

**Procedure**

On Qiskit

1. We start off with the Ideal Qiskit circuit, run the qiskit model iteratively for the total number of entanglement generations over 30 seconds (say n entanglements generated by the source in poisson) in an ideal simulator
2. Measure and record the outcomes and timestamps on qiskit
3. Scale the timestamp values down to 30 seconds

Post processing (detector noise model on both Alice and Bob)

4. Apply efficiency noise model (only 10% incidents detected) over a random sampler to remove 90% of the outcomes
5. Add the dark counts (1000 Per second) 30000 counts extra outcomes (Timestamps are random over uniform distribution)
6. Simulate deadtime and remove outcomes that follow within 4 microseconds

Results

7. Compare coincidences and calculate total count rate, coincidence rate, fidelity, and Bell inequality

## Installation

**System used for testing**

* OS: Ubuntu 20.04 LTS/22.04 LTS
* Python: 3.10.6

**Prerequisites and Dependencies**

* Python3
* numpy
* qiskit
* pip

**Installation and Run**

* Upgrade/update: `sudo apt-get update`
* Install pip: `sudo apt install python3-pip`
* Create a new folder: `mkdir Quantum`
* cd into the folder: `cd Quantum`
* Clone the git repo to find the code: `git clone https://github.ncsu.edu/kgowda/Quantum-Comm-Project.git`
* cd into the project folder: `cd Quantum-Comm-Project`
* Install dependencies: `pip3 install -r requirements.txt` / `pip install -r requirements.txt`
* Run full simulation including Qiskit measurement generation: `python3 Qiskit.py 1` / `python Qiskit.py 1`
* Run simulation using pre-computed measurements for faster processing: `python3 Qiskit.py 2` / `python Qiskit.py 2`
* View the results on the command line

**Note**: Since we are running 30 * 15000 simulations, and then calculating coincidences for such high number of observations, the time taken to run the entire project is quite large. As such, we have implemented two modes. Mode 1: Full simulation including measurement generation. Mode 2: Simulation using pre-computed measurements. Using mode 2, the results can be computed much faster since the bulk of the time is taken by Qiskit measurements. Mode is specified as the command line argument as mentioned in above steps.

If full simulation has to be run, for testing, decreasing the simulation time and dead count can be considered. For this, change line `93` in `Qiskit.py` file to `SECONDS = <required>`. Also, change line `50` in `error_post.py` file to `dark_vals = np.array(self.generate_dark_count_list(30, base, 100, 1))`. Then, run the above mentioned steps.

Also, note that pre-computed measurements are stored currently for only 30 seconds simulation. If such measurements are required for another time window, first uncomment lines `141-157` in `Qiskit.py`, run the simulation in mode 1, and thereafter run it again in mode 2.

## Directory structure

    .
	├── data
	|	├── 1								# Folder containing pre-computed measurements for all theta values across various measurement basis for case 1. Currently contains data for only 30 seconds window
	|	├── 2								# Folder containing pre-computed measurements for all theta values across various measurement basis for case 2. Currently contains data for only 30 seconds window
	|	├── 3								# Folder containing pre-computed measurements for all theta values across various measurement basis for case 3. Currently contains data for only 30 seconds window
	|	├── 4								# Folder containing pre-computed measurements for all theta values across various measurement basis for case 4. Currently contains data for only 30 seconds window
	├── .gitignore                          				# File for git ignore
    ├── error_post.py							# Code for post processing
    ├── perf.py                          					# Code for computing final results
    ├── Qiskit.py                          					# Code for Qiskit simulation
    ├── README.md                          					# Readme file for the project
    ├── requirements.txt							# Details of dependency packages
	└── results.txt								# Final simulation results during testing
