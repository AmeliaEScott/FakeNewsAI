import tensorflow as tf
import numpy as np
import os
import sys
import json
import psycopg2
from gensim.models import KeyedVectors


# The following variables should all be the same as they are in RNNTest2.py
VARIABLE_SAVE_FILE = "VariableCheckpoints/FakeNewsAIVariables.ckpt"
WORD_VECTOR_SIZE = 300
STATE_SIZE = 3000

WEIGHTS_NAME = 'weights'
BIASES_NAME = 'biases'
VARIABLE_SCOPE = "fakenewsvariablescope"

INPUTS_NAME = 'inputs'
INITIAL_STATE_NAME = 'initialState'
INITIAL_HIDDEN_STATE_NAME = 'initialHiddenState'
OUTPUTS_NAME = 'outputs'
SEQUENCE_LENGTH_NAME = 'sequenceLength'

BATCH_SIZE = 10

# If the network outputs above this threshold, the article is true.
# If below, then false.
TRUE_THRESHOLD = 0.51

# If true, then the output is averaged for every word.
# If false, only the output at the last word is considered.
AVERAGE = True

dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../dbsettings.json")
try:
    with open(configpath) as configFile:
        config = json.load(configFile)
except FileNotFoundError:
    print("Didn't find dbsettings.json in %s" % configpath)
    sys.exit()

try:
    connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                                  password=config["password"], port=config["port"])
    cursor = connection.cursor()
except psycopg2.OperationalError:
    print("Error connecting to database. Check that all of the information in your dbsettings.json is correct.")
    sys.exit()


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

    sequence_length = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE], name=SEQUENCE_LENGTH_NAME)

    with tf.variable_scope(VARIABLE_SCOPE):
        cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, initial_state=initial_state_tuple, inputs=inputs,
                                                     sequence_length=sequence_length)

        weights = tf.get_variable(name=WEIGHTS_NAME, shape=[STATE_SIZE, 1], dtype=tf.float32)
        biases = tf.get_variable(name=BIASES_NAME, shape=[1], dtype=tf.float32)

    rnn_outputs_reshaped = tf.reshape(rnn_outputs, shape=[-1, STATE_SIZE])
    network_outputs = tf.reshape(tf.sigmoid(tf.matmul(rnn_outputs_reshaped, weights) + biases), shape=[BATCH_SIZE, -1],
                                 name=OUTPUTS_NAME)

    return inputs, initial_state, initial_hidden_state, network_outputs


def getdata(model):
    cursor.execute("SELECT an.content, an.num_words, s.valid FROM articles_normalized an "
                   "JOIN sources s ON an.source=s.url WHERE an.testing_set ORDER BY num_words ASC;")
    result = cursor.fetchmany(BATCH_SIZE)
    while result is not None and len(result) == BATCH_SIZE:
        # print("Num words: ")
        # print(str([article[1] for article in result]))
        timesteps = max([article[1] for article in result])
        inputs = np.zeros(shape=[BATCH_SIZE, timesteps, WORD_VECTOR_SIZE])
        numwords = np.zeros(shape=[BATCH_SIZE], dtype=np.int32)
        valid = np.zeros(shape=[BATCH_SIZE], dtype=np.bool)
        for articlenum in range(0, BATCH_SIZE):
            numwords[articlenum] = result[articlenum][1]
            valid[articlenum] = True if result[articlenum][2] == 'true' else False
            words = result[articlenum][0].split(" ")
            for wordnum in range(0, len(words)):
                if words[wordnum] in model:
                    inputs[articlenum, wordnum] = model[words[wordnum]]

        yield inputs, numwords, valid
        result = cursor.fetchmany(BATCH_SIZE)


inputs, initial_state, initial_hidden_state, network_outputs = buildgraph()

saver = tf.train.Saver()

print("Loading language model...")
dir = os.path.dirname(__file__)
modelpath = os.path.join(dir, "GoogleNews-vectors-negative300.bin")
# This step takes a VERY LONG TIME (1-2 minutes) and LOTS of memory (5 GB), so only do it once in any program!
try:
    model = KeyedVectors.load_word2vec_format(modelpath, binary=True)
except FileNotFoundError:
    print("Could not load the language file. Make sure you've downloaded it to "
          "AITests/GoogleNews-vectors-negative300.bin. See the README for where to download this from.")
    sys.exit()
print("Done loading language model")

with tf.Session() as session:
    saver.restore(session, VARIABLE_SAVE_FILE)

    articlecount = 0

    # Number of articles correctly identified as true
    truepositives = 0

    # Number of articles incorrectly identified as true
    falsepositives = 0

    # Number of articles correctly identified as false
    truenegatives = 0

    # Number of articles incorrectly identified as false
    falsenegatives = 0

    for input_values, numwords, valid in getdata(model):
        output_results = session.run([network_outputs], feed_dict={
            INPUTS_NAME + ':0': input_values,
            INITIAL_STATE_NAME + ':0': np.zeros(shape=[BATCH_SIZE, STATE_SIZE], dtype=np.float32),
            INITIAL_HIDDEN_STATE_NAME + ':0': np.zeros(shape=[BATCH_SIZE, STATE_SIZE], dtype=np.float32),
            SEQUENCE_LENGTH_NAME + ':0': numwords
        })
        # for i in range(0, len(output_results)):
            # print("Expected: %d. Actual: %s." % (1 if valid[i] else 0, str(output_results[i])))
        for i in range(0, len(valid)):
            articlecount += 1
            correctanswer = valid[i]
            if AVERAGE:
                actualanswer = sum(output_results[0][i][0:numwords[i]]) / numwords[i]
            else:
                actualanswer = output_results[0][i][-1]

            if correctanswer:
                if actualanswer > TRUE_THRESHOLD:
                    truepositives += 1
                else:
                    falsenegatives += 1
            else:
                if actualanswer < TRUE_THRESHOLD:
                    truenegatives += 1
                else:
                    falsepositives += 1

        print("So far, it's gotten %d right out of %d total. That's %.1f%% correct."
              % (truepositives + truenegatives, articlecount, ((truepositives + truenegatives) / articlecount) * 100))
        print("%.1f%% true positives, %.1f%% true negatives, %.1f%% false positives, %.1f%% false negatives."
              % ((truepositives / articlecount) * 100, (truenegatives / articlecount) * 100,
                 (falsepositives / articlecount) * 100, (falsenegatives / articlecount) * 100))
