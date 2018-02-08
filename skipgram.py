import cupy as np 
import sklearn
import math
import sys
import argparse
from operator import itemgetter
import copy

class DataProcessor:
	def remove_new_line(self, file, out):
		with open(file, 'r') as f, open(out, 'w+') as g:
			for line in f:
				temp =  line.replace('\n', ' ')
				g.write(temp)
		print ("Created merged dataset...\n")

	def compute_vocab(self, file):
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
		#Sort the vocabulary
		sorted_vocab = sorted(vocab.items(), key = itemgetter(1))
		#Also compute probabilities in text
		prob_vocab =  {}
		no_vocab={}
		for key, value in sorted_vocab:
			#print ("%s %s" % (key, value))
			prob_vocab[key] = (math.sqrt( 10000* float(value)/j ) + 1) * (float(1.0)/(10000* float(value)/j))
			no_vocab[key] = float(value)
		#Sort the probability vocabulary dictionary
		sorted_prob_vocab =  sorted(prob_vocab.items(), key = itemgetter(1))
		print ()
		print ("Unique: ",len(vocab))
		print ("Total words: ", j)
		return prob_vocab, no_vocab

	def make_training_data(self, file, prob_vocab,no_vocab,n=3 ):
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

		#Make word to integer encoding
		int_to_words={}
		words_to_int = {}
		x=0
		for i, val in enumerate(data_raw):
			if (val[0] in words_to_int):
				continue
			#x+=1
			words_to_int[val[0]] = x
			int_to_words[x] = val[0]
			x+=1
		print ("Unique after removing: ", x)

		return words_to_int, int_to_words, data_raw


class SkipGram:
	def __init__(self, words_to_int, int_to_words, X_train, Y_train,  lr=0.01, dim=300, epochs=100, print_metrics=True):
		
		#np.random.seed(1332)
		self.words_to_int = copy.deepcopy(words_to_int)
		self.model = copy.deepcopy(words_to_int)
		self.int_to_words = copy.deepcopy(int_to_words)
		self.X_train =  copy.deepcopy(X_train)
		self.Y_train = copy.deepcopy(Y_train)
		self.vocab_size = len(words_to_int)
		self.lr = copy.deepcopy(lr) #Learning rate
		self.dim = copy.deepcopy(dim) #Dimensions for our trained vectors
		self.epochs = copy.deepcopy(epochs)
		self.w_hidden = np.random.randn(self.vocab_size, self.dim)
		self.w_output = np.random.randn(self.dim, self.vocab_size)
		#self.onehot = onehot
		#Uninitialised model as of now
		

	def sigmoid(self, theta):

		return (1.0/ (1+ np.exp(-theta)))

	def softmax(self, theta):
		#No need to specify axis since it is 1xV dim
		return (np.exp(theta - np.max(theta)) / np.sum(np.exp(theta- np.max(theta)), axis = 0))

	def one_hot(self, n):
		temp = np.zeros((self.vocab_size, 1))
		temp[n] =1
		return temp
		#return (np.eye(self.vocab_size)[n]).reshape(self.vocab_size,1)

	def build_skipgram_model(self):
		#Iterate over epochs
		print ("No. of training samples are: ", len(self.X_train))
		for k in range(self.epochs):
			print ("We are at epoch : ", k)
			print ()
			print ("No. of training samples: ", len(self.X_train))
			#For each training example
			for i in range(len(self.X_train)/20):

				#Forward propagation of the SkipGram network-----
				#Here X_train[i] is a Vx1 vector.
				#print "self.X_train[i] is ", self.X_train[i]
				#print "self.words_to_int[i] is ", self.words_to_int[self.X_train[i]]

				h = np.dot(self.w_hidden.T , self.one_hot(self.words_to_int[self.X_train[i]]))
				output = np.dot(self.w_output.T , h)
				pred = self.softmax(output)
				print ("---------------")
				print ("Forward propagation done SKIPGRAM...",  str(i)+"/"+str(len(self.X_train)), " Epoch: ", str(k+1)+"/"+str(self.epochs))

				#Backward propagation------
				err_sum = np.zeros((self.vocab_size,1))

				for word in self.Y_train[i]:
					err_sum += (pred - self.one_hot(self.words_to_int[word]))

				#err_sum/= self.vocab_size
				print ("Calculated error.." , i, k+1)


				#Calculate dL/dW
				dw_hidden = np.outer(self.one_hot(self.words_to_int[self.X_train[i]]), np.dot(self.w_output,err_sum))

				#Calculate dL/dW'
				dw_output = np.outer(h, err_sum)

				#Gradient descent
				self.w_hidden += -self.lr * dw_hidden
				self.w_output += -self.lr * dw_output

				print ("Gradient descent done.." , i, k+1)

			#Update model after each epoch
			print ("Saving model...")
			for key, value in self.int_to_words.items():
				self.model[value] = self.w_hidden[key].reshape(1, self.w_hidden.shape[1])

			#Store model after every epoch
			#if (k!k%2==0):	
			print ("Model to npy file...")
			np.save('./utils/skipgram_'+str(k), self.model)

def train(inp, out, dimensions, lr, win, epochs):

	#Preprocess the file
	processor = DataProcessor()
	prob_vocab , no_vocab = processor.compute_vocab(inp)
	words_to_int, int_to_words, data_raw = processor.make_training_data(inp,  prob_vocab, no_vocab,  win)

	
	#Create training data
	X = []
	Y = []
	
	for i, val in enumerate(data_raw):
		X.append(val[0])
		Y.append(val[1:])


	model = SkipGram( words_to_int, int_to_words, X, Y,  lr, dimensions, epochs, True)
	model.build_skipgram_model()



parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Training file', dest='inp', default = './data/')
parser.add_argument('-m','--model', help='Output model file', dest='out')
parser.add_argument('-d', '--dim', help='Dimensionality of word embeddings', dest='dimensions', default=300, type=int)
parser.add_argument('-r', '--rate', help='Learning rate', dest='lr', default=0.025, type=float)
parser.add_argument('-w', '--window', help='Max window length', dest='win', default=3, type=int) 
parser.add_argument('-e','--epochs', help='Number of training epochs', dest='epochs', default=1, type=int)
args = parser.parse_args()

train(args.inp, args.out, args.dimensions, args.lr, args.win, args.epochs)

