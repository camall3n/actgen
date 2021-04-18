import os
import csv
import math
import argparse
import logging

from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)


def read_csv(fname):
	"""
	used to read in gscore csv & reward csv
	"""
	with open(fname, 'r') as f:
		reader = csv.reader(f)
		next(reader, None)  # skip over the header of csv
		reader_results = [[float(i) for i in row] for row in list(reader)]
		columns = list(zip(*reader_results))
	return np.array(columns)


def preprocess_g_score(directory):
	"""
	go through the specified directory and compute the data needed for plotting
	return time_step, avg_g_diff, confidence_interval
	"""
	if not os.path.exists(directory):
		logging.info("directory {} not found".format(directory))
		return [], [], []
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
			g_difference = np.expand_dims(plus_g - minus_g, 0) if len(g_difference) == 0 else np.vstack([g_difference, plus_g - minus_g])
	# average the g_difference and get 95% confidence interval
	# each row of g_difference contains an example
	if len(g_difference) == 0:
		raise RuntimeWarning("no training gscore csv files found in the specified directory")
	avg_g_difference = np.mean(g_difference, axis=0)
	ci = 1.96 * np.std(g_difference, axis=0) / math.sqrt(len(g_difference))
	return time_step, avg_g_difference, ci


def preprocess_training_rewards(directory):
	"""
	go through the specified directory and compute the data needed for plotting
	return time_step, rewards, confidence_interval
	"""
	if not os.path.exists(directory):
		logging.info("directory {} not found".format(directory))
		return [], [], []
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
	return time_step, avg_rewards, ci


def plot_training_data(normal, oracle, no_duplicate, exp_name, data_type):
	"""
	plot the normal DQN and oracle DQN results on the same graph
	used to plot training g-score or reward
	"""
	plt.figure()
	plt.plot(normal[0], normal[1], label='normal {}'.format(data_type))
	plt.fill_between(normal[0], normal[1] - normal[2], normal[1] + normal[2], alpha=.1, label="normal 95% CI")
	# draw oracle only if one was provided
	if len(oracle[0]) != 0:
		plt.plot(oracle[0], oracle[1], label='oracle {}'.format(data_type))
		plt.fill_between(oracle[0], oracle[1] - oracle[2], oracle[1] + oracle[2], alpha=.1, label="oracle 95% CI")
	# draw original no duplication environment only if one was provided
	if len(no_duplicate[0]) != 0:
		plt.plot(no_duplicate[0], no_duplicate[1], label='no duplicate {}'.format(data_type))
		plt.fill_between(no_duplicate[0], no_duplicate[1] - no_duplicate[2], no_duplicate[1] + no_duplicate[2], alpha=.1, label="no duplicate 95% CI")
	# plt.ylim((-1, 1))
	plt.title('{} over time during training with {}'.format(data_type, exp_name))
	plt.xlabel('training step')
	plt.ylabel('{}'.format(data_type))
	plt.legend()
	plt.show()


def plot_all_learning_curves(results_dir, tag, param_tuned):
	"""
	plot all the learning curves for different hyperparams
	"""
	plt.figure()
	for subdir in os.listdir(results_dir):
		if subdir.startswith(tag + param_tuned):
			absolute_dir = results_dir + subdir
			rewards = preprocess_training_rewards(absolute_dir)
			plt.plot(rewards[0], rewards[1], label=subdir)
			plt.fill_between(rewards[0], rewards[1] - rewards[2], rewards[1] + rewards[2], alpha=.1)
	plt.title('learninge curves for all {}'.format(tag + param_tuned))
	plt.xlabel('training step')
	plt.ylabel('reward')
	plt.legend()
	plt.show()


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--results_dir', type=str, default='./results/',
						help='Path to the result directory of saved results files')
	parser.add_argument('--tag', type=str, default='default_exp',
						help='a subdirectory name for the saved results')
	parser.add_argument('--tune_lr', default=False, action='store_true',
                            help='plot all training curves of different learning rate on one graph')
	args = parser.parse_args()
	return args


def main():
	# parse arguments
	args = parse_args()
	directory = args.results_dir + args.tag

	# learning curve to tune learning rate
	if args.tune_lr:
		plot_all_learning_curves(args.results_dir, args.tag, param_tuned='-lr')
		return

	# preprocess the csv
	normal_exp_gscore = preprocess_g_score(directory)
	oracle_exp_gscore = preprocess_g_score(directory + "-oracle")
	no_duplicate_exp_gscore = preprocess_g_score(directory + '-nodup')
	
	normal_exp_reward = preprocess_training_rewards(directory)
	oracle_exp_reward = preprocess_training_rewards(directory + "-oracle")
	no_duplicate_exp_reward = preprocess_training_rewards(directory + "-nodup")

	# plot 
	plot_training_data(normal_exp_gscore, oracle_exp_gscore, no_duplicate_exp_gscore, args.tag, "g difference")
	plot_training_data(normal_exp_reward, oracle_exp_reward, no_duplicate_exp_reward, args.tag, "rewards")


if __name__ == '__main__':
	main()
