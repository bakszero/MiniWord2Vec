import sys
#import keras 
from operator import itemgetter
import math
import numpy as np
import cupy as cp
from cupy.linalg import norm
import sys

def remove_new_line(file, out):
	with open(file, 'r') as f, open(out, 'w+') as g:
		for line in f:
			temp =  line.replace('\n', ' ')
			g.write(temp)
	print ("Created merged dataset...\n")



def compute_vocab(file):
	j = 0
	vocab={}
	text = []
	with open(file, 'r') as f:
		for line in f:
			text = line.split() #Since only 1 line exists

	for word in text:
		j+=1
		if word in vocab:
			vocab[word]+=1
			continue
			#print (word, vocab[word])

		vocab[word] = 1


	sorted_vocab = sorted(vocab.items(), key = itemgetter(1))
	#Also compute probabilities in text
	prob_vocab =  {}
	no_vocab={}
	for key, value in sorted_vocab:
		#print ("%s %s" % (key, value))

		prob_vocab[key] = (math.sqrt( 10000* float(value)/j ) + 1) * (float(1.0)/(10000* float(value)/j))
		no_vocab[key] = float(value)


	sorted_prob_vocab =  sorted(prob_vocab.items(), key = itemgetter(1))
	#for key, value in sorted_prob_vocab:
		#print ("%s %s" % (key, value))


	print ()


	print ("Unique: ",len(vocab))
	print ("Total words: ", j)

	return prob_vocab, no_vocab

def cosine_similarity(x, y):
	a = x.reshape((x.shape[1],))
	b = y.reshape((y.shape[1],))
	return cp.inner(a,b) / norm(a)*norm(b)


def find_similar(file, word):
	doc = np.load(file)

	vec = doc.item()
	word_vec = vec.get(word)
	dist = {}
	print type(vec)
	for element in vec.items():
		#print element[0]
		
		if (element[0] == word):
			continue

		arr = element[1]
		cos = cosine_similarity(word_vec, arr) 
		dist[element[0]] = cos
		print element[0] , dist[element[0]]

	sorted_dist  = sorted(dist.items(), key =  itemgetter(1))
	print (sorted_dist)




def make_training_data(file, prob_vocab,no_vocab,n=3 ):

	text = []
	with open(file ,'r') as f:
		for line in f:
			text = line.split()

	
	data_raw = []

	l=0
	for i in range(n+1, len(text)-n):
		if (no_vocab[text[i]] <  15):
			continue
		l+=1
		temp_context = [text[j] for j in range(i-n, i+n+1) if (j!=i and no_vocab[text[j]] >= 15)]
		temp_context.insert(0, text[i])
		data_raw.append(temp_context)

	print ("Length after removing: ", len(data_raw))



	'''
	with open('./data/skipgram.tsv','w+') as o:
		for key, value in context.items():
			o.write('%s\t' % (key))
			for i, word in enumerate(value):
				o.write('%s' % (word))
				if (i!=(len(value)-1)):
					o.write(',')
			o.write('\n')
			


	with open('./data/cbow.tsv','w+') as o:
		for key, value in context.items():
			for i, word in enumerate(value):
				o.write('%s' % (word))
				if (i!=(len(value)-1)):
						o.write(',')
			o.write('\t%s' % (key))
			o.write('\n')

	'''
	#Make word to integer encoding
	int_to_words={}
	words_to_int = {}
	x=0


	for i, val in enumerate(data_raw):
		if (val[0] in words_to_int):
			continue
		x+=1
		words_to_int[val[0]] = x 	
		int_to_words[x] = val[0]

	print ("Unique after removing: ", x)

	#PC hangs
	#Make one-hot encoding
	#one_hot= {}
	#nb_labels = len(int_to_words)


	'''
	for key, value in words_to_int.items():
		one_hot[key] = np.eye(nb_labels)[value].reshape(nb_labels,1)
		#one_hot[key] =  np.zeros((len(int_to_words), 1))

		#one_hot[key] = [0 for _ in range(len(int_to_words))]
		#one_hot[key][value] = 1
		print (key, "encoded", one_hot[key].shape)

	print (one_hot['of'])
	'''
	return words_to_int, int_to_words, data_raw
			



#remove_new_line(sys.argv[1], sys.argv[2])


#prob_vocab , no_vocab = compute_vocab(sys.argv[1])

#make_training_data(sys.argv[1],  prob_vocab, no_vocab,  3)

find_similar(sys.argv[1], sys.argv[2])