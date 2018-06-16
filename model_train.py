from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from pandas import DataFrame
from matplotlib import pyplot
import pickle
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

#load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    #load doc
    doc = load_doc(filename)
    #clean doc
    tokens = clean_doc(doc)
    #update counts
    vocab.update(tokens)

#load the document
filename = "review_polarity/txt_sentoken/pos/cv000_29590.txt"
text = load_doc(filename)
tokens = clean_doc(text)
print("Tokens\n")
print(tokens)
 
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)
 
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

def process_docs1(directory, vocab):
    #walk through all files in the folder
    for filename in listdir(directory):
        #skip any reviews in the test set
        if filename.startswith('cv9'):
            continue
        #create the full path of the file to open
        path = directory + '/' + filename
        #add doc to vocab
        add_doc_to_vocab(path, vocab)

#save list to file
def save_list(lines, filename):
    #convert lines to a single blob of text
    data = '\n'.join(lines)
    #open file
    file = open(filename,'w')
    #write text
    file.write(data)
    #close file
    file.close()

# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	scores = list()
	n_repeats = 10
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network
		model = Sequential()
		model.add(Dense(10, input_shape=(n_words,), activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# compile network
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit network
		model.fit(Xtrain, ytrain, epochs=10, verbose=2)
		# evaluate
		loss, acc = model.evaluate(Xtest, ytest, verbose=0)
		scores.append(acc)
		print('%d accuracy: %s' % ((i+1), acc))
	return scores

# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
	# create the tokenizer
	tokenizer = Tokenizer(num_words=25768)
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest

# classify a review as negative (0) or positive (1)
def predict_sentiment(review, vocab, tokenizer, model):
	# clean
	tokens = clean_doc(review)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	# convert to line
	line = ' '.join(tokens)
	# encode
	encoded = tokenizer.texts_to_matrix([line], mode='freq')
	# prediction
	yhat = model.predict(encoded, verbose=0)
	return round(yhat[0,0])

#define vocab
vocab = Counter()
#add all docs to vocab
process_docs1("review_polarity/txt_sentoken/pos", vocab)
process_docs1("review_polarity/txt_sentoken/neg", vocab)
#print the size of the vocab
print("Length of Vocab\n")
print(len(vocab))
#print the top words in the vocab
print("50 most common words\n")
print(vocab.most_common(50))

#keep tokens with a min occurence
min_occurence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurence]
print("Length of tokens\n")
print(len(tokens))

#save tokens to a vocabulary file
save_list(tokens,'vocab.txt')


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_lines = process_docs('review_polarity/txt_sentoken/pos', vocab, True)
negative_lines = process_docs('review_polarity/txt_sentoken/neg', vocab, True)
train_docs = negative_lines + positive_lines

#summarize what we have
print("Length of positive lines\n")
print(len(positive_lines))
print("Length of negative lines\n")
print(len(negative_lines))


# create the tokenizer
tokenizer = Tokenizer(num_words=25768)
# fit the tokenizer on the documents
docs = negative_lines + positive_lines
tokenizer.fit_on_texts(docs)

# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
 
# load all test reviews
positive_lines = process_docs('review_polarity/txt_sentoken/pos', vocab, False)
negative_lines = process_docs('review_polarity/txt_sentoken/neg', vocab, False)
test_docs = negative_lines + positive_lines

# prepare labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# encode training data set
Xtest = tokenizer.texts_to_matrix(test_docs, mode='freq')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
print("Shape of Xtest\n")
print(Xtest.shape)
print("Shape of ytest\n")
print(ytest.shape)
 
n_words = Xtest.shape[1]
# define network
model = Sequential()
model.add(Dense(10, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

modes = ['binary', 'count', 'tfidf', 'freq']
# modes = ['binary']
results = DataFrame()
for mode in modes:
	# prepare data for mode
	Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
	# evaluate model on data for mode
	results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest)



# summarize results
print("Summarize results\n")
print(results.describe())
# plot results
results.boxplot()
pyplot.show()

# test positive text
positive_filename = 'positive.txt'
pos_text = load_doc(positive_filename)
print(pos_text)
print("Prediction\n")
print(predict_sentiment(pos_text, vocab, tokenizer, model))

# test negative text
negative_filename = 'negative.txt'
neg_text = load_doc(negative_filename)
print(neg_text)
print("Prediction\n")
print(predict_sentiment(neg_text, vocab, tokenizer, model))