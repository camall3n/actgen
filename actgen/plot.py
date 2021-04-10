import os
import csv
import math
import argparse

from matplotlib import pyplot as plt
import numpy as np


def read_csv(fname):
	"""
	used to read in gscore csv & reward csv
	"""
	with open(fname, 'r') as f:
		reader = csv.reader(f)
		reader_results = [[float(i) for i in row] for row in list(reader)]
		columns = list(zip(*reader_results))
	return np.array(columns)


def plot_training_g_score(directory, tag):
	time_step = np.array([])
	g_difference = np.array([])
	# iterate over the directory
	for file_name in os.listdir(directory):
		# find all the saved gscore files
		if file_name.endswith("training_gscore.csv"):
			file_path = os.path.join(directory, file_name)
			step, plus_g, minus_g = read_csv(file_path)
			if len(time_step) == 0:
				time_step = step
			assert time_step.all() == step.all()
			g_difference = plus_g - minus_g if len(g_difference) == 0 else np.vstack([g_difference, plus_g - minus_g])
	# average the g_difference and get 95% confidence interval
	# each row of g_difference contains an example
	if len(g_difference) == 0:
		raise RuntimeWarning("no training gscore csv files found in the specified tag directory")
	avg_g_difference = np.mean(g_difference, axis=0)
	ci = 1.96 * np.std(g_difference, axis=0) / math.sqrt(len(g_difference))

	# plot
	plt.figure()
	plt.plot(time_step, avg_g_difference, label='average g difference')
	plt.fill_between(time_step, avg_g_difference - ci, avg_g_difference + ci, alpha=.1, label="95% CI")
	# plt.ylim((-1, 1))
	plt.title(f'g difference over time during training with {tag}')
	plt.xlabel('training step')
	plt.ylabel('g diffence')
	plt.legend()
	plt.show()


def plot_training_rewards(directory, tag):
	time_step = np.array([])
	rewards = np.array([])
		# iterate over the directory
	for file_name in os.listdir(directory):
		# find all the saved gscore files
		if file_name.endswith("training_reward.csv"):
			file_path = os.path.join(directory, file_name)
			step, r = read_csv(file_path)
			if len(time_step) == 0:
				time_step = step
			assert time_step.all() == step.all()
			rewards = r if len(rewards) == 0 else np.vstack([rewards, r])
	# average the g_difference and get 95% confidence interval
	# each row of g_difference contains an example
	if len(rewards) == 0:
		raise RuntimeWarning("no training reward csv files found in the specified tag directory")
	avg_rewards = np.mean(rewards, axis=0)
	ci = 1.96 * np.std(rewards, axis=0) / math.sqrt(len(rewards))

	# plot
	plt.figure()
	plt.plot(time_step, avg_rewards, label='average training reward')
	plt.fill_between(time_step, avg_rewards - ci, avg_rewards + ci, alpha=.1, label="95% CI")
	# plt.ylim((-1, 1))
	plt.title(f'training reward over time during callback with {tag}')
	plt.xlabel('training step')
	plt.ylabel('average reward')
	plt.legend()
	plt.show()


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--results_dir', type=str, default='./results/',
						help='Path to the result directory of saved results files')
	parser.add_argument('--tag', type=str, default='default_exp',
						help='a subdirectory name for the saved results')
	args = parser.parse_args()
	return args


def main():
	# parse arguments
	args = parse_args()
	directory = args.results_dir + args.tag

	# plot g score
	plot_training_g_score(directory, args.tag)

	# plot rewards
	plot_training_rewards(directory, args.tag)


if __name__ == '__main__':
	main()
