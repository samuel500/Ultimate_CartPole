import tensorflow as tf
import numpy as np
import gym
import random
import time

env = gym.make("CartPole-v1")
print("act_spa", env.action_space)
print("env.observation_space", env.observation_space.low)


x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float')


rand_bool = lambda x: x*100 > random.randrange(0, 100) #Returns True with probability x


def neural_network(data):

	n_nodes_1 = 16
	#n_nodes_2 = 16
	n_outputs = 2
	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([4, n_nodes_1])),
		'biases':tf.Variable(tf.random_normal([n_nodes_1]))}
	#hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_1, n_nodes_2])),
	#	'biases':tf.Variable(tf.random_normal([n_nodes_2]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_1, n_outputs])),
		'biases':tf.Variable(tf.random_normal([n_outputs]))}

	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	#l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	#l2 = tf.nn.relu(l2)

	output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
	output = tf.nn.softmax(output)
	return output


def test_NN(NN, iterations = 100, render = False, verbose = False):

	startT = time.time()

	scores = []
	for i in range(iterations):

		env.reset()

		observation, reward, done, info = env.step(env.action_space.sample())
		reward_tot = 0
		for s in range(500):
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
		print("Time:" + str(totT))
		print("Mean score:", np.mean(scores))

	return scores


def generate_samples(nb_samples = 10000, min_score = 50, neg_mem = 8, render = False, p_random = 1, NN = None, verbose = False, auto = False):

	if verbose and not auto:
		print("Generate_samples")
		print("nb_ samples:", nb_samples)
		print("min_score:", min_score)
		print("neg_mem:", neg_mem)
		print("p_random", p_random)
		print("auto", str(auto))

	samples = [[], []]

	all_scores = []

	if p_random < 1 and NN == None:
		print("Warning: probability of non-random action less than 1 and no NN model given.")
	'''
	if auto and NN != None:
		meanNN = test_NN(NN, render = render, verbose = verbose)
		print("meanNN:", meanNN)

		nb_samples = 100000
		min_score = meanNN + 30
		neg_mem = 12

		p_random = 1-meanNN/300
		if p_random < 0.12:
			p_random = 0.12

		if verbose and auto:
			print("Generate_samples")
			print("nb_ samples:", nb_samples)
			print("min_score:", min_score)
			print("neg_mem:", neg_mem)
			print("p_random", p_random)
			print("auto", str(auto))
	'''

	tot_samples = 0
	for i in range(nb_samples):
		tot_samples += 1
		env.reset()

		prev_obs, reward, done, info = env.step(env.action_space.sample())
		reward_tot = 0
		interm_samples = [[], []]
		for s in range(500):
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
			interm_samples[0].append(np.array(prev_obs))
			interm_samples[1].append(np.array(action_rep))

			prev_obs = observation

			if done:
				all_scores.append(reward_tot)
				break

		if reward_tot >= min_score:
			#samples[0].append(interm_samples[0][:-neg_mem-1])
			#samples[1].append(interm_samples[1][:-neg_mem-1])
			samples[0] += interm_samples[0][:-neg_mem-1]
			samples[1] += interm_samples[1][:-neg_mem-1]

			#Test
			if len(samples[0]) > 40000 and auto:
				break

	if verbose:
		print("tot_samples", tot_samples)
		print("samples gen:", len(samples[0]))
	#print(samples)
	assert len(samples[0]) == len(samples[1])
	return samples



def train(x):
	p_random = 1 #Start with only random actions
	negative_memory = 4

	NN_action = neural_network(x)
	cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions = NN_action, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		verbose = True
		nb_epochs = 8 #"epochs"
		batch_size = 16

		training_episodes = [ #Is not very consistent
			{'samples': {'nb_samples': 500, 'min_score': 25, 'p_random': 1, 'neg_mem': 4, 'verbose': verbose, 'NN': NN_action},
				'nb_epochs': 8, 'batch_size': 16},
			{'samples': {'nb_samples': 500, 'min_score': 180, 'p_random': 0.5, 'neg_mem': 12, 'verbose': verbose, 'NN': NN_action},
				'nb_epochs': 8, 'batch_size': 16},
			{'samples': {'nb_samples': 500, 'min_score': 400, 'p_random': 0.18, 'neg_mem': 16, 'verbose': verbose, 'NN': NN_action},
				'nb_epochs': 8, 'batch_size': 32},
		]


		for episode in training_episodes:

			samples = generate_samples(**episode['samples'])


			for e in range(episode['nb_epochs']):
				print("epoch", e)

				for b in range(int(len(samples[0])/episode['batch_size'])):
					#print(np.array(memory[0][b*batch_size:]))
					_, c = sess.run([optimizer, cost], feed_dict = {x: np.array(samples[0][b*episode['batch_size']:(b+1)*episode['batch_size']]), y: np.array(samples[1][b*episode['batch_size']:(b+1)*episode['batch_size']])})


			test_list = test_NN(NN = NN_action, iterations = 40, render = False, verbose = False)
			print("Mean:", np.mean(test_list))

		test_NN(NN = NN_action, iterations = 100, render = True, verbose = True)


train(x)
