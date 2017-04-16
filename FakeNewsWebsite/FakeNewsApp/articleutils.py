"""
This file does all of the actual work of parsing an article
and inputting it to the neural network.
"""
import requests
from newspaper import fulltext
import re
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
import os


# The following variables should all be the same as they are in RNNTest2.py
VARIABLE_SAVE_FILE = "VariableCheckpoints/FakeNewsAIVariables.ckpt"
WORD_VECTOR_SIZE = 300
STATE_SIZE = 3000
TRUNCATION = 200000000

WEIGHTS_NAME = 'weights'
BIASES_NAME = 'biases'
VARIABLE_SCOPE = "fakenewsvariablescope"

INPUTS_NAME = 'inputs'
INITIAL_STATE_NAME = 'initialState'
INITIAL_HIDDEN_STATE_NAME = 'initialHiddenState'
OUTPUTS_NAME = 'outputs'
SEQUENCE_LENGTH_NAME = 'sequenceLength'

BATCH_SIZE = 1

dir = os.path.dirname(__file__)
modelpath = os.path.join(dir, "GoogleNews-vectors-negative300.bin")

# This step takes a VERY LONG TIME (1-2 minutes) and LOTS of memory (5 GB), so only do it once in any program!
print("Loading language model... (This will take 1-2 minutes)")
model = KeyedVectors.load_word2vec_format(modelpath, binary=True)
print("Done loading language model.")


def getarticletext(url):
    """
    Downloads the HTML of an article, and extracts the body text.
    :param url: URL of article to download
    :return: String containing the entire text of the article, with newlines intact, or None if invalid URL
    """
    response = requests.get(url)
    if response is None:
        return None
    html = requests.get(url).text
    if html is None:
        return None
    text = fulltext(html)
    text = re.sub("\n+", "\n", text)
    return text


def texttovector(text):
    """
    Normalizes the text by doing the following:
    1. Remove all punctuation, except apostrophes
    2. Replace all whitespace with spaces
    3. Replace any multiple consecutive spaces with single spaces
    4. Convert all words to lowercase
    5. Replace all digits with #
    6. Replace any words not present in the language model with <UNK>

    Then, converts the normalized text to its vector representation using the language model.

    :param text: Text to normalize
    :return: Tuple of (length, article): Length is the number of words in the article. Article is
             a numpy array of shape [1, length, 300]
    """

    text = text.lower()

    # Replace all whitespace with single spaces
    text = re.sub("\s+", " ", text)

    # Replace all digits with number sign
    text = re.sub("\d", "#", text)

    # Remove anything that's not a letter, number sign, apostrophe, or space
    text = re.sub("[^a-z#' ]", "", text, flags=re.IGNORECASE)

    words = text.split(" ")

    totalwords = len(words)

    result = np.zeros(shape=[BATCH_SIZE, totalwords, WORD_VECTOR_SIZE])

    for word, num in zip(words, range(0, totalwords)):
        if word in model:
            result[0, num] = model[word]
    return totalwords, result


def buildgraph():
    """
    Builds the graph, without adding the training step.
    :return: (inputs, initial_state, initial_hidden_state, network_outputs)
        inputs: A placeholder for the sequence of inputs to the network
        initial_state: A placeholder for the initial state for the RNN
        initial_hidden_state: A placeholder for the initial hidden state for the RNN
        network_outputs: The tensor representing the outputs from the network
    """

    inputs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, WORD_VECTOR_SIZE], name=INPUTS_NAME)
    initial_state = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, STATE_SIZE], name=INITIAL_STATE_NAME)
    initial_hidden_state = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, STATE_SIZE],
                                          name=INITIAL_HIDDEN_STATE_NAME)

    initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(initial_state, initial_hidden_state)

    # sequence_length = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE], name=SEQUENCE_LENGTH_NAME)

    with tf.variable_scope(VARIABLE_SCOPE):
        cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, initial_state=initial_state_tuple, inputs=inputs)

        weights = tf.get_variable(name=WEIGHTS_NAME, shape=[STATE_SIZE, 1], dtype=tf.float32)
        biases = tf.get_variable(name=BIASES_NAME, shape=[1], dtype=tf.float32)

    rnn_outputs_reshaped = tf.reshape(rnn_outputs, shape=[-1, STATE_SIZE])
    network_outputs = tf.reshape(tf.sigmoid(tf.matmul(rnn_outputs_reshaped, weights) + biases), shape=[BATCH_SIZE, -1],
                                 name=OUTPUTS_NAME)

    return final_state, network_outputs


def scorearticle(textvector, numwords):
    """
    Inputs the network to the neural net and returns the average of the score.
    :param textvector: Article text converted to its vector representation, as returned by texttovector()
    :return: A list of numbers representing the score at each word of the article, and a single number
            representing the average score over the entire article.
    """

    # results = np.zeros(shape=[numwords])
    state = np.zeros(shape=[BATCH_SIZE, STATE_SIZE], dtype=np.float32)
    hiddenstate = np.zeros(shape=[BATCH_SIZE, STATE_SIZE], dtype=np.float32)
    # for start in range(0, numwords, TRUNCATION):
    # end = min(start + TRUNCATION, numwords)
    finalstateresult, outputsresult = session.run([finalstate, outputs], feed_dict={
        INPUTS_NAME + ":0": textvector,
        INITIAL_STATE_NAME + ":0": state,
        INITIAL_HIDDEN_STATE_NAME + ":0": hiddenstate,
    })
    outputsresult = outputsresult[0]
    # results[start:end] = outputsresult
    # state = finalstateresult[0]
    # hiddenstate = finalstateresult[1]
    return outputsresult, sum(outputsresult) / len(outputsresult)


print("Initializing TensorFlow graph...")
finalstate, outputs = buildgraph()
saver = tf.train.Saver()
session = tf.Session()
variablespath = os.path.join(dir, VARIABLE_SAVE_FILE)
saver.restore(session, variablespath)
print("Done initializing graph.")

if __name__ == "__main__":
    print("Scoring article...")
    text = getarticletext("https://www.washingtonpost.com/world/national-security/trump-officials-tell-russia-to-drop-its-support-for-syrias-assad/2017/04/09/c179d3ba-4713-440b-8192-a2838019554d_story.html?hpid=hp_hp-top-table-main_tillerson-645pm%3Ahomepage%2Fstory&utm_term=.f3f098cb5de4")
    length, vector = texttovector(text)
    score = scorearticle(vector, length)
    print("Final score: %.6f" % score[-1])