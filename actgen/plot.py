import os
import csv
import math
import argparse

from matplotlib import pyplot as plt
import numpy as np

def read_g_score_csv(fname):
	with open(fname, 'r') as f:
		reader = csv.reader(f)
		g_scores = list(reader)
		time_step = [int(i[0]) for i in g_scores]
		plus_g = [float(i[1]) for i in g_scores]
		minus_g = [float(i[2]) for i in g_scores]
	return np.array(time_step), np.array(plus_g), np.array(minus_g)


def plot_training_g_score(directory, tag):
	time_step = np.array([])
	g_difference = np.array([])
	# iterate over the directory
	for file_name in os.listdir(directory):
		# find all the saved gscore files
		if file_name.endswith("training_gscore.csv"):
			file_path = os.path.join(directory, file_name)
			step, plus_g, minus_g = read_g_score_csv(file_path)
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

	# plot g score
	directory = args.results_dir + args.tag
	plot_training_g_score(directory, args.tag)


if __name__ == '__main__':
	main()
