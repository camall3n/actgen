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
		# only gather 20k steps for pendulum
		if 'pendulum' in dirname:
			# TODO:
			data_frames = [pd.read_csv(fname) for fname in all_files]
			filtered_data_frames = [df[df['training step'] <= 20000] for df in data_frames]
			data_in_dir = pd.concat(filtered_data_frames, ignore_index=True)
			# data_in_dir = pd.concat([pd.read_csv(fname) for fname in all_files], ignore_index=True)
		else:
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


env_id_2_env_name = {
	"cartpole-dqn": "CartPole",
	"pendulum-dqn": "Pendulum",
	"lander-dqn": "LunarLander",
	"beam-rider": "Beam Rider",
	"breakout": "Breakout",
	"pacman": "Ms Pacman",
	"space-invaders": "Space Invaders",
	"qbert": "Qbert",
	"pong": "Pong",
}


def plot_reward(data, env_name, save_path=None):
	"""
	plot the reward results of all the agent types from a single gym environment
	"""
	# sns.set_theme()
	ax = sns.relplot(
		data=data, kind="line",
		x='training step', y='reward during evaluation callback',
		hue='agent', 
		style='agent',
	)
	plot_env_name = env_id_2_env_name[env_name]
	plt.title('{}'.format(plot_env_name))
	plt.ylabel('Episode Reward')
	plt.xlabel('Training Steps')
	ax._fig.subplots_adjust(top=0.9)
	ax._fig.subplots_adjust(bottom=0.12)
	plt.legend(loc='best')
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
	parser.add_argument('--plot_random', default=False, action='store_true',
							help='plot the training curve of all random actions experiments')
	parser.add_argument('--plot_duplicate', default=False, action='store_true',
							help='plot the training curve of all experiments comparing DQN with DQN on N-dup')
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
		experiment_dir + "-nodup": "1x (baseline)",
		# experiment_dir: "5 dup",
		# experiment_dir + "-oracle": "5 dup with oracle",
	}
	if args.plot_ndup:
		dirname_to_description[experiment_dir] = "5x"
		dirname_to_description[experiment_dir + "-oracle"] = "5x (oracle)"
		all_values_of_n = [15, 50]
		for n in all_values_of_n:
			dirname_to_description[experiment_dir + "-{}dup".format(n)] = "{}x".format(n)
			dirname_to_description[experiment_dir + "-{}dup-oracle".format(n)] = "{}x (oracle)".format(n)
	elif args.plot_semi:
		all_semi_scores = [0.2, 0.5, 0.8]
		for k in all_semi_scores:
			dirname_to_description[experiment_dir + "-semi-{}".format(k)] = "5x, h={}".format(k)
			dirname_to_description[experiment_dir + "-semi-{}-oracle".format(k)] = "5x, h={} (oracle)".format(k)
	elif args.plot_atari:
		args.suppress_gscore = True  # atari doesn't have gscore
		games_to_plot = ['breakout', 'pacman', 'space-invaders', 'qbert', 'pong', 'beam-rider']
		fig, axes = plt.subplots(2, 3, sharex=True)
		for i, game in enumerate(games_to_plot):
			# gather data from csv
			experiment_dir = args.results_dir + game	
			dirname_to_description = {
				experiment_dir: "Baseline (1x)",
				experiment_dir + "-5dup": "Duplicate (5x)",
				experiment_dir + "-full": "Full action set",
				experiment_dir + "-more": "Noop (2x)",
			}
			data = preprocess_data("training_reward.csv", dirname_to_description)
			# plot is with subplot
			sns.lineplot(
				ax=axes[i // 3, i % 3],
				data=data,
				x='training step', y='reward during evaluation callback',
				hue='agent', style='agent'
			)
			axes[i // 3, i % 3].set_title(f"{env_id_2_env_name[game]}")
			# ylabel
			if i % 3 == 0:
				axes[i // 3, i % 3].set_ylabel("Episode Reward")
			else:
				axes[i // 3, i % 3].set_ylabel("")
			axes[i//3, i%3].set_xlabel("Training Steps")
			# shared legend
			axes[i // 3, i % 3].get_legend().remove()
			if i == 5:
				handles, labels = axes[i // 3, i % 3].get_legend_handles_labels()
				fig.legend(handles, labels, loc='lower center', ncol=4)
				plt.subplots_adjust(bottom=0.15)
		plt.show()
		plt.close()
		return
	elif args.plot_duplicate:
		args.suppress_gscore = True
		envs_to_plot = ['cartpole-dqn', 'pendulum-dqn', 'lander-dqn']
		fig, axes = plt.subplots(1, 3)
		for i, env in enumerate(envs_to_plot):
			# gather data from csv
			experiment_dir = args.results_dir + env
			dirname_to_description = {
				experiment_dir + "-nodup": "1x (baseline)",
				experiment_dir: "5x",
				experiment_dir + "-oracle": "5x (oracle)",
			}
			data = preprocess_data("training_reward.csv", dirname_to_description)
			# plot is with subplot
			sns.lineplot(
				ax=axes[i],
				data=data,
				x='training step', y='reward during evaluation callback',
				hue='agent', style='agent'
			)
			axes[i].set_title(f"{env_id_2_env_name[env]}")
			fig.subplots_adjust(bottom=0.12)
			# ylabel
			if i == 0:
				axes[i].set_ylabel("Episode Reward")
			else:
				axes[i].set_ylabel("")
			axes[i].set_xlabel("Training Steps")
			# shared legend
			axes[i].get_legend().remove()
			if i == 2:
				handles, labels = axes[i].get_legend_handles_labels()
				fig.legend(handles, labels, loc='lower center', ncol=4)
				plt.subplots_adjust(bottom=0.20)
		plt.show()
		plt.close()
		return
	elif args.plot_random:


		args.suppress_gscore = True
		envs_to_plot = ['cartpole-dqn', 'pendulum-dqn', 'lander-dqn']
		fig, axes = plt.subplots(1, 3)
		for i, env in enumerate(envs_to_plot):
			# gather data from csv
			experiment_dir = args.results_dir + env
			dirname_to_description = {
				experiment_dir + "-nodup": "1x (baseline)",
				experiment_dir + "-rand": "5x",
				experiment_dir + "-rand-oracle": "5x (oracle)",
			}
			data = preprocess_data("training_reward.csv", dirname_to_description)
			# plot is with subplot
			sns.lineplot(
				ax=axes[i],
				data=data,
				x='training step', y='reward during evaluation callback',
				hue='agent', style='agent'
			)
			axes[i].set_title(f"{env_id_2_env_name[env]}")
			fig.subplots_adjust(bottom=0.12)
			# ylabel
			if i == 0:
				axes[i].set_ylabel("Episode Reward")
			else:
				axes[i].set_ylabel("")
			axes[i].set_xlabel("Training Steps")
			# shared legend
			axes[i].get_legend().remove()
			if i == 2:
				handles, labels = axes[i].get_legend_handles_labels()
				fig.legend(handles, labels, loc='lower center', ncol=4)
				plt.subplots_adjust(bottom=0.20)
		plt.show()
		plt.close()
		return


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
