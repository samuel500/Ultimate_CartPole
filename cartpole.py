import tensorflow as tf
import numpy as np
import gym
import random
import time
import itertools
import pickle

env = gym.make("CartPole-v1")
print("act_spa", env.action_space)
print("env.observation_space", env.observation_space.low)


x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float')


nb_games_used = 0

rand_bool = lambda x: x*100 > random.randrange(0, 100) #Returns True with probability x



def neural_network(data):

	#p_dropout = 0.2
	n_nodes_1 = 16
	#n_nodes_2 = 16
	n_outputs = 2
	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([4, n_nodes_1]), name = 'weights_1'),
		'biases':tf.Variable(tf.random_normal([n_nodes_1]), name = 'biases_1')}
	#hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_1, n_nodes_2])),
	#	'biases':tf.Variable(tf.random_normal([n_nodes_2]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_1, n_outputs]), name = 'weights_out'),
		'biases':tf.Variable(tf.random_normal([n_outputs]), name = 'biases_out')}

	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)
	#l1 = tf.nn.dropout(l1, p_dropout)

	#l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	#l2 = tf.nn.relu(l2)

	output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
	output = tf.nn.softmax(output)
	return output



def test_NN(NN, iterations = 4, render = False, verbose = False):

	startT = time.time()

	scores = []
	for i in range(iterations):

		env.reset()

		observation, reward, done, info = env.step(env.action_space.sample())
		reward_tot = 0
		for s in range(600):
			if render:
				env.render()

			nn_input = {x: [observation]}
			action = NN.eval(nn_input)
			action = np.argmax(action[0])

			observation, reward, done, info = env.step(action)
			reward_tot += 1

			if done:
				break
		if verbose:
			print("reward_tot:", reward_tot)
		scores.append(reward_tot)


	totT = time.time() - startT

	if verbose:
		print("test_NN(iterations=" + str(iterations) + ")")
		print("Time:" + str(totT) + " s")
		print("Mean score:", np.mean(scores))

	return scores



def generate_samples(nb_games = 5000, min_score = 50, neg_mem = 8, min_samples = 0, max_samples = 200000,  render = False, p_random = 1, NN = None, verbose = False, auto = False):

	global nb_games_used
	if verbose:
		print("Generate_samples")
		#print("nb_ games:", nb_games)
		print("min_score:", min_score)
		print("neg_mem:", neg_mem)
		print("p_random", p_random)

	samples = []

	all_scores = [] #?

	if p_random < 1 and NN == None:
		print("Warning: probability of non-random action less than 1 and no NN model given.")

	tot_games = 0
	for i in range(nb_games):
		tot_games += 1
		env.reset()

		prev_obs, reward, done, info = env.step(env.action_space.sample())
		reward_tot = 0
		interm_samples = []
		for s in range(600):
			if render:
				env.render()

			if rand_bool(p_random) or NN == None:
				action = env.action_space.sample()
			else:
				nn_input = {x: [prev_obs]}
				action = NN.eval(nn_input)
				action = np.argmax(action[0])

			observation, reward, done, info = env.step(action)

			reward_tot += reward

			action_rep = [0, 0]
			action_rep[action] = 1
			interm_samples.append(np.append(np.array(prev_obs), np.array(action_rep)))

			prev_obs = observation

			if done:
				all_scores.append(reward_tot)
				break

		if reward_tot >= min_score:
			samples.append(interm_samples[:-neg_mem-1])

			#Test
			if sum([len(game) for game in samples]) > max_samples:
				break

	if verbose:
		print("tot_games", tot_games)
		print("samples gen:", len(samples))
		print("Mean score gen:", np.mean(all_scores))

	nb_games_used += tot_games
	#print(samples)
	return samples #[ [np.array([obs1, obs2, obs3, obs4, action1, action2]), ... ], ... ]



def train(x, target_score = 480, save = False):
	global nb_games_used
	p_random = 1 #Start with only random actions
	negative_memory = 4

	NN_action = neural_network(x)
	cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions = NN_action, labels=y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		verbose = True
		min_epochs = 8
		max_epochs = 24 #"epochs"
		batch_size = 16

		samples = []
		min_samples = 1000
		max_samples = 1000
		max_nb_samples = 60000
		min_score = 22
		neg_mem = 5
		p_random = 1
		nb_samples = 0
		sample_drop_threshold = 600000
		sample_drop_rate = 0

		sample_drop_threshold_worst = 200000
		sample_drop_rate_worst = 0.0
		iter_count = 0
		while np.mean(test_NN(NN_action, iterations = 40)) < target_score:
			print("iter_count:", iter_count)
			iter_count += 1
			samples_opt = {'min_score': min_score, 'max_samples': max_samples, 'p_random': p_random, 'neg_mem': neg_mem, 'NN': NN_action, 'verbose': verbose}


			samples = [sample for sample in samples if len(sample) > min_score] #Filter out all samples that are too short
			#samples = []
			samples += generate_samples(**samples_opt)
			print([len(game) for game in samples])
			nb_samples = sum([len(game) for game in samples])

			print("nb_games_used:", nb_games_used)


			if nb_samples > sample_drop_threshold:
				random.shuffle(samples)
				samples = samples[int(len(samples)*sample_drop_rate):]

			if nb_samples > sample_drop_threshold_worst:
				samples.sort(key = len)
				samples = samples[int(len(samples)*sample_drop_rate_worst):]

			if nb_samples > max_nb_samples:
				samples.sort(key = len, reverse = True)
				cutoff = 0
				count_samples = 0
				for i in range(len(samples)):
					count_samples += len(samples[i])
					if count_samples > max_nb_samples:
						cutoff = i
						break

				samples = samples[:i]
				print("lensamples?:", len(samples))
				nb_samples = sum([len(game) for game in samples])
				print("nb_samples:", nb_samples)
			random.shuffle(samples)


			train_samples = np.array(list(itertools.chain.from_iterable(samples)))

			try:
				X = train_samples[:, :-2]
				Y = train_samples[:, -2:]
			except IndexError:
				print("len train_samples:", len(train_samples))
				raise
			print("Curr nb samples:", len(X))

			prev_mean = 0

			for e in range(max_epochs):
				print("epoch", e)

				for b in range(int(len(X)/batch_size)):
					#print(np.array(memory[0][b*batch_size:]))
					_, c = sess.run([optimizer, cost], feed_dict = {x: np.array(X[b*batch_size:(b+1)*batch_size]), y: np.array(Y[b*batch_size:(b+1)*batch_size])})


				test_list = test_NN(NN = NN_action, iterations = 100, render = False, verbose = False)
				curr_mean = np.mean(test_list)
				print("Mean:", curr_mean)
				if (prev_mean > curr_mean and e > min_epochs-1) or curr_mean > target_score:
					print("Peak training at epoch", e)
					break
				prev_mean = curr_mean

			#Update params (?):
			min_score += 12
			if min_score < 30:
				min_score = 30
			max_samples *= 2
			if max_samples > 80000:
				max_samples = 80000
			p_random *= 0.92
			if  p_random < 0.15:
				p_random = 0.15
			neg_mem *= 2
			if neg_mem > 12:
				neg_mem = 12
			neg_mem = int(neg_mem)
			test_NN(NN = NN_action, iterations = 4, render = True, verbose = True)

		print("Final mean:", np.mean(test_NN(NN = NN_action, iterations = 100)))
		test_NN(NN = NN_action, iterations = 8, render = True, verbose = True)
		if save:
			saver = tf.train.Saver()
			saver.save(sess, './cartpole.ckpt')

			print("Saved")




def save_NN(NN, name = "cartpole.pkl"):

	with open(name, 'wb') as output:
		pickle.dump(NN, output)

	print("Saved successfully")


def load_NN(name = "cartpole.ckpt", test = False):

	NN = neural_network(x)
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, './' + name)
		if test:
			print("Mean:", np.mean(test_NN(NN, iterations = 100)))
			test_NN(NN, iterations = 8, render = True, verbose = True)

	return NN


if __name__ == '__main__':

	games_used_list = []
	try:
		load_NN(test = True)
	except tf.errors.NotFoundError:
		print("No trained model found.")
		train(x, save = True, target_score = 480)

