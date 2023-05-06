import numpy as np
import random
from perf import CoincidenceMonitor

class PostProcessor:
	'''
	Post processor class for Bell Inequality Test simulation
	'''
	def __init__(self, list_alice, list_bob, density_matrix) -> None:
		'''
		Parse input arrays, and simulate error models
		'''
		self.bases = [0, 1]
		self.lists = [np.array(list_alice), np.array(list_bob)]
		self.count_rates = {
			'alice_0': 0,
			'alice_1': 0,
			'bob_0': 0,
			'bob_1': 0
		}
		self.alice_0 = []
		self.alice_1 = []
		self.bob_0 = []
		self.bob_1 = []
		self.coincidence_rates = {
			'00': 0,
			'01': 0,
			'10': 0,
			'11': 0
		}
		self.fidelity = 0
		self.density_matrix = density_matrix
		self.compute_metrics()
		
	def compute_metrics(self):
		'''
		Post process and compute metrics
		'''
		for base in self.bases:
			is_alice = True
			for l in self.lists:
				# Filter out list with required base
				base_list = []
				for val in l:
					if val[1] == base:
						base_list.append(val)
				# Get 10% values from the input
				filtered_vals = np.array(self.pick_vals(base_list))
				# Get dark counts at the rate of 1000 Hz
				dark_vals = np.array(self.generate_dark_count_list(30, base, 1000, 10))
				# Concatenate and sort both filtered and dark count values
				total_vals = np.concatenate((filtered_vals, dark_vals))
				total_vals = sorted(total_vals, key=lambda x: x[0])
				# Simulate dead time
				reported_vals = self.simulate_dead_time(total_vals)
				# Store values based on user and base, and compute count rates
				if is_alice:
					if base == 0:
						self.alice_0 = np.array(reported_vals)
						self.count_rates['alice_0'] = CoincidenceMonitor().count_rate(self.alice_0, 1)
					else:
						self.alice_1 = np.array(reported_vals)
						self.count_rates['alice_1'] = CoincidenceMonitor().count_rate(self.alice_1, 1)
					is_alice = False
				else:
					if base == 0:
						self.bob_0 = np.array(reported_vals)
						self.count_rates['bob_0'] = CoincidenceMonitor().count_rate(self.bob_0, 1)
					else:
						self.bob_1 = np.array(reported_vals)
						self.count_rates['bob_1'] = CoincidenceMonitor().count_rate(self.bob_1, 1)
					is_alice = True
		# Compute coincidence rates
		self.coincidence_rates['00'] = CoincidenceMonitor(reported_alice=self.alice_0, reported_bob=self.bob_0).coincidence_rate()
		self.coincidence_rates['01'] = CoincidenceMonitor(reported_alice=self.alice_0, reported_bob=self.bob_1).coincidence_rate()
		self.coincidence_rates['10'] = CoincidenceMonitor(reported_alice=self.alice_1, reported_bob=self.bob_0).coincidence_rate()
		self.coincidence_rates['11'] = CoincidenceMonitor(reported_alice=self.alice_1, reported_bob=self.bob_1).coincidence_rate()
		self.total_coincidence_rates = {}
		self.total_coincidence_rates['00'] = self.coincidence_rates['00']['true'][0] + self.coincidence_rates['00']['false'][0]
		self.total_coincidence_rates['01'] = self.coincidence_rates['01']['true'][0] + self.coincidence_rates['01']['false'][0]
		self.total_coincidence_rates['10'] = self.coincidence_rates['10']['true'][0] + self.coincidence_rates['10']['false'][0]
		self.total_coincidence_rates['11'] = self.coincidence_rates['11']['true'][0] + self.coincidence_rates['11']['false'][0]
		self.fidelity = CoincidenceMonitor().fidelity(self.total_coincidence_rates, self.density_matrix)


	def pick_vals(self, arr):
		'''
		Filter input to get 10% values
		'''
		# Randomly sample 10% values
		selected_indices = random.sample(range(len(arr)), int(0.1 * len(arr)))
		selected_vals = []
		for index in selected_indices:
			selected_vals.append(arr[index])
		return selected_vals

	def generate_dark_count_list(self, on_time=30, base=0, per_second_average=1000, per_second_std_deviation=10):
		'''
		Generate dark count values at the average rate of 1000 Hz
		'''
		list_dark_count = []
		for t in range(on_time):
			# Get dark count timestamps
			per_second_dark_count = int(np.floor(np.random.normal(per_second_average, per_second_std_deviation, 1)))
			dark_count_timestamps = np.random.uniform(t, t + 1, per_second_dark_count)
			for timestamp in dark_count_timestamps:
				list_dark_count.append([timestamp, base, 1])
		return list_dark_count

	def simulate_dead_time(self, filtered_vals, dead_time=0.000004):
		'''
		Simulate 4 microseconds dead time
		'''
		n = len(filtered_vals)
		if n > 0:
			result = [filtered_vals[0]]
			curTime = filtered_vals[0][0]
			i = 1
			while i < n:
				if curTime + dead_time < filtered_vals[i][0]:
					result.append(filtered_vals[i])
					curTime = filtered_vals[i][0]
				i += 1
			return result
		return []
