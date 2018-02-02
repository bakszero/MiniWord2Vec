import sys
#import keras 
from operator import itemgetter

def remove_new_line(file, out):
	with open(file, 'r') as f, open(out, 'w+') as g:
		for line in f:
			temp =  line.replace('\n', ' ')
			g.write(temp)



def do_something(file):
	vocab={}
	text = []
	with open(file, 'r') as f:
		for line in f:
			text = line.split() #Since only 1 line exists

	for word in text:
		if word in vocab:
			vocab[word]+=1
			continue
			#print (word, vocab[word])

		vocab[word] = 1

	return vocab
#remove_new_line(sys.argv[1], sys.argv[2])
vocab = do_something(sys.argv[1])

print (len(vocab))

sorted_vocab = sorted(vocab.items(), key = itemgetter(1))
for key, value in sorted_vocab:
	print ("%s %s" % (key, value)	)
print (len(vocab))