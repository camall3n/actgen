import os
import argparse
import logging

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


logging.basicConfig(level=logging.INFO)


def read_csv(fname, agent):
    data = pd.read_csv(fname)
    data['agent'] = agent
    return data


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


def preprocess_data(experiment_name, file_ending):
	"""
	get all the csv data from a single gym environment experiment from all types of agents
	return a pandas Dataframe object
	"""
		# gather all files from agent
	n_dup_agent = experiment_name
	n_dup_files = iterate_through_directory(n_dup_agent, file_ending)

	n_dup_oracle_agent = experiment_name + "-oracle"
	n_dup_oracle_files = iterate_through_directory(n_dup_oracle_agent, file_ending)

	no_dup_agent = experiment_name + "-nodup"
	no_dup_files = iterate_through_directory(no_dup_agent, file_ending)

	rand_actions_agent = experiment_name + "-rand"
	rand_actions_files = iterate_through_directory(rand_actions_agent, file_ending)

	rand_actions_oracle_agent = experiment_name + "-rand-oracle"
	rand_actions_oracle_files = iterate_through_directory(rand_actions_oracle_agent, file_ending)

	# gather all csv data from the files
	fnames = n_dup_files + n_dup_oracle_files + no_dup_files + rand_actions_files + rand_actions_oracle_files
	agents = ["N dup"] * len(n_dup_files) + \
				["N dup with oracle"] * len(n_dup_oracle_files) + \
				["no dup"] * len(no_dup_files) + \
				["random actions"] * len(rand_actions_files) + \
				["random actions with oracle"] * len(rand_actions_oracle_files)
	data = pd.concat([read_csv(fname, agent) for fname, agent in zip(fnames, agents)])

	return data


def gather_data_for_param_tuning(results_dir, tag, param_tuned):
	"""
	plot all the learning curves for different hyperparams
	"""
	fnames = []
	hyperparams = []
	for subdir in os.listdir(results_dir):
		if subdir.startswith(tag + '-' + param_tuned):
			absolute_dir = results_dir + subdir
			logging.info('found directory for tuning {}: {}'.format(param_tuned, absolute_dir))
			files_in_dir = iterate_through_directory(absolute_dir, "training_reward.csv")
			fnames += files_in_dir
			hyperparams += [subdir] * len(files_in_dir)
	data = pd.concat([read_csv(fname, hyperparam) for fname, hyperparam in zip(fnames, hyperparams)])
	return data


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


def plot_reward(data, env_name):
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
	parser.add_argument('--plot_loss', default=False, action='store_true',
                        	help='plot the training loss')
	parser.add_argument('--suppress_gscore', default=False, action='store_true',
							help='do not plot gscore')
	args = parser.parse_args()
	return args


def main():
	# parse arguments
	args = parse_args()
	expriment_dir = args.results_dir + args.tag

	# learning curve to tune learning rate
	if args.tune_lr:
		tune_data = gather_data_for_param_tuning(args.results_dir, args.tag, param_tuned='lr')
		plot_reward(tune_data, args.tag)
		return

	# plot reward
	reward_data = preprocess_data(expriment_dir, "training_reward.csv")
	plot_reward(reward_data, args.tag)

	# plot g score
	if not args.suppress_gscore:
		g_score_data = preprocess_data(expriment_dir, "training_gscore.csv")
		plot_g_score(g_score_data, args.tag)

	# plot training loss
	if args.plot_loss:
		loss_data = preprocess_data(expriment_dir, "training_loss.csv")
		plot_batch_loss(loss_data, args.tag)


if __name__ == '__main__':
	main()
