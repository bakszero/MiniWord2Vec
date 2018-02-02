import sys
#import keras 
from operator import itemgetter
import math

def remove_new_line(file, out):
	with open(file, 'r') as f, open(out, 'w+') as g:
		for line in f:
			temp =  line.replace('\n', ' ')
			g.write(temp)



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
	for key, value in sorted_vocab:
		print ("%s %s" % (key, value)	)

		prob_vocab[key] = (math.sqrt( 10000* float(value)/j ) + 1) * (float(1.0)/(10000* float(value)/j))


	sorted_prob_vocab =  sorted(prob_vocab.items(), key = itemgetter(1))
	for key, value in sorted_prob_vocab:
		print ("%s %s" % (key, value))


	print ()


	print ("Unique: ",len(vocab))
	print ("Total words: ", j)



	


#remove_new_line(sys.argv[1], sys.argv[2])

compute_vocab(sys.argv[1])


