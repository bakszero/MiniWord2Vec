import numpy as np 
import sklearn
import sys
import argparse




class SkipGram:
	def __init__(self, onehot, X_train, Y_train,  lr=0.01, dim=300, epochs=100, print_metrics=True):
		
		np.random.seed(1332)
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
		for k in range(self.epochs):
			print ("We are at epoch : ", k)

			#For each training example
			for i in range(self.vocab_size):

				#Forward propagation of the neural network-----
				#Here X_train[i] is a Vx1 vector.
				h = np.dot(self.w_hidden.T , self.onehot[X_train[i]])
				output = np.dot(self.w_output.T , h)
				pred = self.softmax(output)

				#Backward propagation------
				err_sum = np.zeros((self.vocab_size,1))

				for word in self.Y_train[i]:
					err_sum += (pred - self.onehot(word))

				#err_sum/= self.vocab_size

				#Calculate dL/dW
				dw_hidden = np.outer(self.onehot(X_train[i]), np.dot(self.w_output,err_sum))

				#Calculate dL/dW'
				dw_output = np.outer(h, err_sum)

				#Gradient descent
				self.w_hidden += -self.lr * dw_hidden
				self.w_output += -self.lr * dw_output





def train(inp, out, dimensions, lr, win, epochs):

	#Preprocess the file
	#Create vocab and one-hot


	X = []
	Y = []
	with open (inp , 'r') as f:
		for line in f:
			X.append(line.split('\t')[0])
			Y.append((line.split('\t')[1]).split(','))

	model = SkipGram()





if name == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training file', dest='inp', required=True)
    parser.add_argument('--model', help='Output model file', dest='out', required=True)
    parser.add_argument('--dim', help='Dimensionality of word embeddings', dest='dimensions', default=300, type=int)
    parser.add_argument('--rate', help='Learning rate', dest='lr', default=0.025, type=float)
    parser.add_argument('--window', help='Max window length', dest='win', default=3, type=int) 
    parser.add_argument('--epochs', help='Number of training epochs', dest='epochs', default=1, type=int)
    args = parser.parse_args()

    train(args.inp, args.out, args.dimensions, args.lr, args.win, args.epochs)

