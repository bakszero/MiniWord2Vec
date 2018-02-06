import cupy as np 
import sklearn
import math
import sys
import argparse
from operator import itemgetter

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
			x+=1
			words_to_int[val[0]] = x
			int_to_words[x] = val[0]

		print ("Unique after removing: ", x)

		return words_to_int, int_to_words, data_raw


class CBoW:
	def __init__(self, words_to_int, int_to_words, X_train, Y_train,  lr=0.01, dim=300, epochs=100, print_metrics=True):
		
		#Input will be flipped for skipgram, but we maintain 2 different classes for cleaner comparison of the models later!
		np.random.seed(1332)
		self.words_to_int = words_to_int
		self.int_to_words = int_to_words
		self.model = words_to_int
		self.X_train =  X_train
		self.Y_train = Y_train
		self.vocab_size = len(words_to_int)
		self.lr = lr #Learning rate
		self.dim = dim #Dimensions for our trained vectors
		self.epochs = epochs
		self.w_hidden = np.random.randn(self.vocab_size, self.dim)
		self.w_output = np.random.randn(self.dim, self.vocab_size)
		#self.onehot = onehot
		#Uninitialised model as of now
		#self.model= {}

	def sigmoid(self, theta):

		return (1.0/ (1+ np.exp(-theta)))

	def softmax(self, theta):
		#No need to specify axis since it is vx1 dim , but do it nevertheless.
		return (np.exp(theta - np.max(theta)) / np.sum(np.exp(theta- np.max(theta)), axis = 0))


	def one_hot(self, n):
		return (np.eye(self.vocab_size)[n]).reshape(self.vocab_size,1)
	
	def build_cbow_model(self):
		#Iterate over epochs
		for k in range(self.epochs):
			print ("We are at epoch : ", k+1)
			#For each training example
			for i in range(self.vocab_size):

				#Forward propagation of the neural network-----

				#Take average
				x = np.zeros((self.vocab_size, 1))
				for word in self.X_train[i]:
					x += self.one_hot(self.words_to_int[word])
				x/= len(self.X_train[i])

				h = np.dot(self.w_hidden.T, x)
				'''
				h = np.zeros((self.dim, 1))
				for word in self.X_train[i]:
					h += np.dot(self.w_hidden.T, self.one_hot(self.words_to_int[word]))
				h/=len(self.X_train[i])
				print ("Forward propagation done...",  i, k)

				'''
				print ("-----------")
				print ("Forward propagation done...: ", i, "Epoch: ", k) 
				#h = np.dot(self.w_hidden.T , onehot(X_train[i])
				u = np.dot(self.w_output.T , h)
				pred = self.softmax(u)

				#Backward propagation------
				#err_sum = np.zeros((self.vocab_size,1))

				err = pred - self.one_hot(self.words_to_int[self.Y_train[i]])
				print ("Calculated error.." , i, k)

				#Calculate dL/dW

				dw_hidden = np.outer(x, np.dot(self.w_output, err))

				#Calculate dL/dW'
				dw_output = np.outer(h,err)

				#Gradient descent
				self.w_hidden += -self.lr * dw_hidden
				self.w_output += -self.lr * dw_output
				print ("Gradient descent done.." , i, k)


			#Update model after each epoch
			print ("Saving model...")
			for key, value in words_to_int.items():
				self.model[key] = self.w_hidden[value].reshape(1, self.w_hidden.shape[1])

			#Store model after every 2 epochs
			if (k!=0 and k%2==0):	
				print ("saveing model...")
				np.save('cbow_'+k, self.model)



def train(inp, out, dimensions, lr, win, epochs):

	#Preprocess the file
	processor  = DataProcessor()
	prob_vocab , no_vocab = processor.compute_vocab(inp)
	words_to_int, int_to_words, data_raw = processor.make_training_data(inp,  prob_vocab, no_vocab,  win)

	
	#Create training data
	X = []
	Y = []
	
	for i, val in enumerate(data_raw):
		X.append(val[1:])
		Y.append(val[0])


	model = CBoW( words_to_int, int_to_words, X, Y,  lr, dimensions, epochs, True)
	model.build_cbow_model(	)






parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Training file', dest='inp', default = './data/')
parser.add_argument('-m','--model', help='Output model file', dest='out')
parser.add_argument('-d', '--dim', help='Dimensionality of word embeddings', dest='dimensions', default=300, type=int)
parser.add_argument('-r', '--rate', help='Learning rate', dest='lr', default=0.025, type=float)
parser.add_argument('-w', '--window', help='Max window length', dest='win', default=3, type=int) 
parser.add_argument('-e','--epochs', help='Number of training epochs', dest='epochs', default=1, type=int)
args = parser.parse_args()

train(args.inp, args.out, args.dimensions, args.lr, args.win, args.epochs)

