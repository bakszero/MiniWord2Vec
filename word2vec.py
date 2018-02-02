import numpy as np 
import sklearn
import sys
import argparse


class DataProcessor:
	def __init__(self, )


class SkipGram:
	def __init__(self, context, onehot, X_train, Y_train,  lr=0.01, dim=300, epochs=100, print_metrics=True):
		
		np.random.seed(1332)
		self.context = context
		self.X_train =  X_train
		self.Y_train = Y_train
		self.vocab_size = X_train.shape[0]
		self.lr = lr #Learning rate
		self.dim = dim #Dimensions for our trained vectors
		self.epochs = epochs
		self.w_hidden = np.random.randn(self.vocab_size, self.dim)
		self.w_output = np.random.randn(self.dim, self.vocab_size)
		self.onehot = onehot
		#Uninitialised model as of now
		self.model= {}

	def sigmoid(self, theta):

		return (1.0/ (1+ np.exp(-theta)))

	def softmax(self, theta):
		#No need to specify axis since it is 1xV dim
		return (np.exp(theta - np.max(theta)) / np.sum(np.exp(theta- np.max(theta)), axis = 0))


	def build_skipgram_model(self):
		#Iterate over epochs
		for _ in range(self.epochs):

			#For each training example
			for i in range(self.vocab_size):

				#Forward propagation of the neural network-----
				#Here X_train[i] is a Vx1 vector.
				h = np.dot(self.w_hidden.T , self.onehot[X_train[i]])
				output = np.dot(self.w_output.T , h)
				pred = self.softmax(output)

				#Backward propagation------
				#http://www.claudiobellei.com/2018/01/06/backprop-word2vec/
				#Calculate EI_j
				err_sum = np.zeros((self.vocab_size,1))

				for word in self.Y_train[i]:
					err_sum += (pred - self.onehot(word))

				#err_sum/= self.vocab_size

				#Calculate dL/dW
				dw_hidden = np.outer(onehot(X_train[i]), np.dot(self.w_output,err_sum))

				#Calculate dL/dW'
				dw_output = np.outer(np.dot(self.w_hidden.T, onehot(X_train[i])), err_sum)

				#Gradient descent
				self.w_hidden += -self.lr * dw_hidden
				self.w_output += -self.lr * dw_output




class CBoW:
	def __init__(self, context, onehot, X_train, Y_train,  lr=0.01, dim=300, epochs=100, print_metrics=True):
		
		#Input will be flipped, but we maintain 2 different classes for cleaner comparison of the models later!
		np.random.seed(1332)
		self.context = context
		self.X_train =  X_train
		self.Y_train = Y_train
		self.vocab_size = X_train.shape[0]
		self.lr = lr #Learning rate
		self.dim = dim #Dimensions for our trained vectors
		self.epochs = epochs
		self.w_hidden = np.random.randn(self.vocab_size, self.dim)
		self.w_output = np.random.randn(self.dim, self.vocab_size)
		self.onehot = onehot
		#Uninitialised model as of now
		self.model= {}

	def sigmoid(self, theta):

		return (1.0/ (1+ np.exp(-theta)))

	def softmax(self, theta):
		#No need to specify axis since it is 1xV dim
		return (np.exp(theta - np.max(theta)) / np.sum(np.exp(theta- np.max(theta)), axis = 0))


	def build_cbow_model(self):
		#Iterate over epochs
		for _ in range(self.epochs):

			#For each training example
			for i in range(self.vocab_size):

				#Forward propagation of the neural network-----
				h = np.zeros((self.dim, 1))
				for word in self.X_train[i]:
					h += np.dot(w_hidden.T, self.onehot[word])
				h/=self.vocab_size


				#h = np.dot(self.w_hidden.T , onehot(X_train[i])
				u = np.dot(self.w_output.T , h)
				pred = self.softmax(u)

				#Backward propagation------
				#http://www.claudiobellei.com/2018/01/06/backprop-word2vec/
				#err_sum = np.zeros((self.vocab_size,1))

				err = pred - self.onehot[Y_train[i]]

				#Calculate dL/dW

				dw_hidden = np.mean([np.outer(word, np.dot(W2, err) for word in self.X_train[i]], axis=0)

				#Calculate dL/dW'
				dw_output = np.outer(h,err)

				#Gradient descent
				self.w_hidden += -self.lr * dw_hidden
				self.w_output += -self.lr * dw_output



def train(inp, out, method, dimensions, lr, win, epochs):

	#Preprocess the file



if name == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training file', dest='inp', required=True)
    parser.add_argument('--model', help='Output model file', dest='out', required=True)
    parser.add_argument('--skipgram', help='1 for skipgram, 0 for CBoW', dest='method', default=1, type=int)
    parser.add_argument('--dim', help='Dimensionality of word embeddings', dest='dimensions', default=300, type=int)
    parser.add_argument('--rate', help='Learning rate', dest='lr', default=0.025, type=float)
    parser.add_argument('--window', help='Max window length', dest='win', default=3, type=int) 
    parser.add_argument('--epochs', help='Number of training epochs', dest='epochs', default=1, type=int)
    args = parser.parse_args()

    train(args.inp, args.out, args.method, args.dimensions, args.lr, args.win, args.epochs)

