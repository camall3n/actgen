import os
import argparse
import logging
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


logging.basicConfig(level=logging.INFO)


def iterate_through_directory(directory, file_ending):
	"""
	go through the specified the specified directory, and look for all the files
	with a specific file_ending
	return a list of file paths
	"""
	if not os.path.exists(directory):
		logging.info("directory {} not found".format(directory))
		return []
	# iterate over the directory
	all_file_paths = []
	for file_name in os.listdir(directory):
		# find all eligible files
		if file_name.endswith(file_ending):
			file_path = os.path.join(directory, file_name)
			all_file_paths.append(file_path)
	return all_file_paths


def preprocess_data(file_ending, dirname_to_description):
	"""
	get all the csv data from a single gym environment experiment from all types of agents
	return a pandas Dataframe object
	args:
		file_ending: the ending format of the file whose data needs to be gathered (e.g. loss.csv)
		dirname_to_descirption: a dictionary mapping the directory name (data that needs to be plotted)
								to the plot's description of that directory of data
	"""
	# gather all the files for each directory
	accumulated_data = []
	for dirname in dirname_to_description:
		description = dirname_to_description[dirname]
		all_files = iterate_through_directory(dirname, file_ending)
		if not all_files:
			continue  # for non-existent directories
		data_in_dir = pd.concat([pd.read_csv(fname) for fname in all_files], ignore_index=True)
		data_in_dir['agent'] = description
		accumulated_data.append(data_in_dir)
	accumulated_data = pd.concat(accumulated_data, ignore_index=True)

	return accumulated_data


def gather_data_for_param_tuning(results_dir, tag, param_tuned):
	"""
	gather all the data/directories needed for plotting the learning curves of 
	different values of the hyperparam being tuned
	"""
	accumulated_data = []
	for subdir in os.listdir(results_dir):
		if subdir.startswith(tag + '-' + param_tuned):
			data_in_subdir = []
			absolute_dir = results_dir + subdir
			logging.info('found directory for tuning {}: {}'.format(param_tuned, absolute_dir))
			files_in_dir = iterate_through_directory(absolute_dir, "training_reward.csv")
			data_in_subdir = pd.concat([pd.read_csv(fname) for fname in files_in_dir])
			data_in_subdir['agent'] = subdir
			accumulated_data.append(data_in_subdir)
	accumulated_data = pd.concat(accumulated_data)
	return accumulated_data


def plot_g_score(data, env_name):
	"""
	plot the g-score results of all the agents from a single gym environment
	"""
	data['g score'] = data['plus_g'].astype(float) - data['minus_g'].astype(float)
	sns.set_theme()
	sns.relplot(
		data=data, kind="line",
		x='training step', y='g score',
		hue='agent', style='agent'
	)
	plt.title('g score of {}'.format(env_name))
	plt.show()


def plot_reward(data, env_name, save_path=None):
	"""
	plot the reward results of all the agent types from a single gym environment
	"""
	sns.set_theme()
	sns.relplot(
		data=data, kind="line",
		x='training step', y='reward during evaluation callback',
		hue='agent', style='agent'
	)
	plt.title('training curve of {}'.format(env_name))
	plt.show()
	if save_path:
		plt.savefig(save_path)
	plt.close()


def plot_batch_loss(data, env_name):
	"""
	plot the training batch loss of all the agent types from a single gym environment
	"""
	sns.set_theme()
	sns.relplot(
		data=data, kind="line",
		x='training step', y='batch loss',
		hue='agent', style='agent'
	)
	plt.title('training loss of {}'.format(env_name))
	plt.show()


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--results_dir', type=str, default='./results/',
						help='Path to the result directory of saved results files')
	parser.add_argument('--tag', type=str, default='default_exp',
						help='a subdirectory name for the saved results')
	parser.add_argument('--tune_lr', default=False, action='store_true',
                            help='plot all training curves of different learning rate on one graph')
	parser.add_argument('--plot_ndup', default=False, action='store_true',
							help='plot the training curve of all N-dup experiments for different values of N')
	parser.add_argument('--plot_semi', default=False, action='store_true',
							help='plot the training curve of all k-semi-dup experiments for different values of k')
	parser.add_argument('--plot_atari', default=False, action='store_true',
							help='plot the training curve of atari experiments and with help of inverse models')
	parser.add_argument('--plot_loss', default=False, action='store_true',
                        	help='plot the training loss')
	parser.add_argument('--suppress_gscore', default=False, action='store_true',
							help='do not plot gscore')
	args = parser.parse_args()
	return args


def main():
	# parse arguments
	args = parse_args()
	experiment_dir = args.results_dir + args.tag

	# learning curve to tune learning rate
	if args.tune_lr:
		tune_data = gather_data_for_param_tuning(args.results_dir, args.tag, param_tuned='lr')
		plot_reward(tune_data, args.tag, save_path= os.path.join(experiment_dir, 'learning_curve.jpg'))
		return
	
	# all data that needs to be plotted
	# base cases needed by all
	dirname_to_description = {
		experiment_dir + "-nodup": "no dup",
		experiment_dir: "5 dup",
		experiment_dir + "-oracle": "5 dup with oracle",
	}
	if args.plot_ndup:
		all_values_of_n = [15, 50]
		for n in all_values_of_n:
			dirname_to_description[experiment_dir + "-{}dup".format(n)] = "{} dup".format(n)
			dirname_to_description[experiment_dir + "-{}dup-oracle".format(n)] = "{} dup with oracle".format(n)
	elif args.plot_semi:
		all_semi_scores = [0.2, 0.5, 0.8, 0.95, 0.99]
		for k in all_semi_scores:
			dirname_to_description[experiment_dir + "-semi-{}".format(k)] = "{} semi-duplicate".format(k)
			dirname_to_description[experiment_dir + "-semi-{}-oracle".format(k)] = "{} semi-duplicate with oracle".format(k)
	elif args.plot_atari:
		args.suppress_gscore = True  # atari doesn't have gscore
		dirname_to_description = {
			experiment_dir: "baseline",
			experiment_dir + "-5dup": "4 sets of duplicate actions",
			experiment_dir + "-full": "full action set (18 actions)",
			experiment_dir + "-more-noop": "2 noop action sets",
			experiment_dir + "-oracle": "similarity oracle",
		}
	else:
		# default option
		dirname_to_description[experiment_dir + "-rand"] = "random actions"
		dirname_to_description[experiment_dir + "-rand-oracle"] = "random actions with oracle"

	# plot reward
	reward_data = preprocess_data("training_reward.csv", dirname_to_description)
	plot_reward(reward_data, args.tag, save_path=os.path.join(experiment_dir, 'learning_curve.jpg'))

	# plot g score
	if not args.suppress_gscore:
		g_score_data = preprocess_data("training_gscore.csv", dirname_to_description)
		plot_g_score(g_score_data, args.tag)

	# plot training loss
	if args.plot_loss:
		loss_data = preprocess_data("training_loss.csv", dirname_to_description)
		plot_batch_loss(loss_data, args.tag)


if __name__ == '__main__':
	main()
