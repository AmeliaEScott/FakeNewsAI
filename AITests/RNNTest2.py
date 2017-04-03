"""
This is a refactoring of RNNTest1.py to use multiple GPUs
"""

import sys
import datetime
import tensorflow as tf
import numpy as np
import os
import math
from gensim.models import KeyedVectors


try:
    from AITests.DataBatching import getbatches
except (ImportError, SystemError):
    try:
        from .AITests.DataBatching import getbatches
    except (ImportError, SystemError):
        print("Could not load data batching module.")
        print("Make sure you run this from the root directory of the repository,")
        print("and use \"python -m AITests.RNNTest1\"")
        sys.exit()

# File in which to save the trained weights and biases
# .ckpt stands for checkpoint
VARIABLE_SAVE_FILE = "VariableCheckpoints/FakeNewsAIVariables.ckpt"

# Size of batch in number of articles
# Batch size of 1 means each article is its own batch
# For the sake of me not having to bugfix, this number
# should be evenly divisible by NUM_GPUS if NUM_GPUS > 0
BATCH_SIZE = 20

# Backpropagation through hundreds of time steps takes waaaay too much memory, so we
# have to limit it. This number should be in the low hundreds, like between 100 and 400
# This is not gauranteed to be the limit of truncation. It will vary by as much as 50%
TRUNCATION = 200

# Every UPDATE_FREQUENCY batches, the variables will be stored to disk, and
# information will be printed about time, loss, etc.
UPDATE_FREQUENCY = 50

# Size of a vector for an individual word
WORD_VECTOR_SIZE = 300

# If True, the network will attempt to get every single output at every time step correct.
# If False, then it will only care about the final output.
SEQ2SEQ = False

# If true, then print out a bunch of debugging stuff in the actual graph
DEBUG = True

# Number of words to input to the network at a time
# TODO: Make sure this number works
WORDS_INPUT_AT_ONCE = 1

# Size of state to remember between iterations within one article
STATE_SIZE = 3000

# Number of GPUs on the target machine. Can be 0
NUM_GPUS = 0

# These three should just remain constant
WEIGHTS_NAME = 'weights'
BIASES_NAME = 'biases'
VARIABLE_SCOPE = "fakenewsvariablescope"

# The learning rate is a constant for how quickly it should learn.
# Too high, and it could overshoot and oscillate around wildly. Too low,
# and it'll just take forever. I have no idea how to find the right value.
LEARNING_RATE = 0.0005


def buildtower(networkinput, initial_state, initial_hidden_state, expected_outputs, loss_mask):
    """
    Builds a single instance of a tower.
    A tower, in this case, is the entire neural network up to the point where it calculates the gradient.
    This is because all of these operations can be run in parallel across multiple GPUs, but the gradient
    must be calculated all at once on the CPU. Thus, the computational graph returned by this function
    can be entirely self-contained within a single GPU.
    :param networkinput: Placeholder for the inputs to the network
    :param initial_state: Placeholder for the initial state for the LSTM
    :param initial_hidden_state: Placeholder for the initial hidden state for the LSTM
    :param expected_outputs: Placeholder for the expected outputs from the network
    :param loss_mask: Placeholder for the loss mask
    :return: The placeholders for the network input, the initial state, and the initial hidden state,
            as well as the loss, loss_mask, and network outputs
    """

    # For much more thorough documentation on this code, see RNNTest1.py.
    # Except where otherwise stated, this code all does roughly the same thing here.

    # States for the LSTM
    initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(initial_state, initial_hidden_state)

    # Create the actual RNN
    try:
        with tf.variable_scope(VARIABLE_SCOPE, reuse=True):
            cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
            rnn_outputs, finalstate = tf.nn.dynamic_rnn(cell=cell, inputs=networkinput,
                                                        initial_state=initial_state_tuple,
                                                        sequence_length=tf.reduce_sum(loss_mask, axis=[1, 2]))
    except ValueError:
        with tf.variable_scope(VARIABLE_SCOPE, reuse=None):
            cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
            rnn_outputs, finalstate = tf.nn.dynamic_rnn(cell=cell, inputs=networkinput,
                                                        initial_state=initial_state_tuple,
                                                        sequence_length=tf.reduce_sum(loss_mask, axis=[1, 2]))

    # Here is where this code differs from RNNTest1. We can't simply construct a new Variable, because
    # these variables must be shared between multiple different towers. This piece of code retrieves
    # Variables that already exist elsewhere in Tensorflow's memory.
    with tf.variable_scope(VARIABLE_SCOPE, reuse=True):
        weights = tf.get_variable(name=WEIGHTS_NAME, shape=[STATE_SIZE, 1], dtype=tf.float32)
        biases = tf.get_variable(name=BIASES_NAME, shape=[1], dtype=tf.float32)

    # Build the output layers
    if SEQ2SEQ:
        rnn_outputs_reshaped = tf.reshape(rnn_outputs, [-1, STATE_SIZE])
    else:
        rnn_outputs_reshaped = tf.reshape(finalstate[0], [-1, STATE_SIZE])
    network_outputs = tf.reshape(tf.sigmoid(tf.matmul(rnn_outputs_reshaped, weights) + biases), shape=[-1])

    # This operation changes the shape of expected_outputs from [BATCH_SIZE] to [BATCH_SIZE, 1],
    # which, as it turns out, is a meaningful distinction.
    # For example, the tensor [1, 2, 3] has shape [3], while the tensor [[1], [2], [3]] has shape [3, 1]
    expected_outputs = tf.expand_dims(expected_outputs, axis=1)

    if SEQ2SEQ:
        # The expected_outputs placeholder is currently shaped [BATCH_SIZE], because it has only one value
        # for each article in the batch. If we are doing seq2seq, then we need to have an expected output
        # for every word of every article. Tile just repeats the tensor for every word.
        # That tf.shape(...) nonsense is because the number of timesteps is unknown (It's the "None" dimension
        # of the inputs), so we have to fetch that number dynamically.
        expected_outputs_tiled = tf.tile(expected_outputs, multiples=[1, tf.shape(networkinput)[1]])
        expected_outputs_reshaped = tf.reshape(expected_outputs_tiled, [-1])
    else:
        expected_outputs_reshaped = tf.reshape(expected_outputs, [-1])

    # network_outputs = loss_mask_reshaped * network_outputs

    if SEQ2SEQ:
        loss_mask_reshaped = tf.reshape(loss_mask, shape=[-1])
    else:
        # If we're not doing Seq2Seq, then there is no loss mask
        loss_mask_reshaped = 1.0

    loss = tf.losses.mean_squared_error(labels=expected_outputs_reshaped, predictions=network_outputs,
                                        weights=loss_mask_reshaped)

    if DEBUG:
        loss = tf.Print(loss, message="Sequence length: ", data=[tf.reduce_sum(loss_mask, axis=[1, 2])],
                        summarize=BATCH_SIZE)
        loss = tf.Print(loss, message="Predictions: ", data=[network_outputs], summarize=BATCH_SIZE)
        loss = tf.Print(loss, message="Expected: ", data=[expected_outputs_reshaped], summarize=BATCH_SIZE)

    return loss, finalstate


# This code is courtesy of the Tensorflow open source examples.
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def averagegradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def buildgraph():
    """
    Builds the computational graph. This constructs NUM_GPUS towers (Or just 1, if NUM_GPUS == 0),
    then creates the gradient averaging and training step.
    :return:
    """

    # This list is the list of each device for which we should make a tower. If there are no GPUs, then there
    # should be exactly 1 tower, on the CPU. If there are 1 or more GPUs, then there should be a tower on
    # each GPU, and none on the CPU.
    gpus = ["/cpu:0"] if NUM_GPUS == 0 else ["/gpu:%d" % i for i in range(0, NUM_GPUS)]

    optimizer = tf.train.AdagradOptimizer(LEARNING_RATE)

    with tf.device("/cpu:0"):
        with tf.variable_scope(VARIABLE_SCOPE):
            tf.get_variable(WEIGHTS_NAME, shape=[STATE_SIZE, 1], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32))
            tf.get_variable(BIASES_NAME, shape=[1], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        network_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, WORD_VECTOR_SIZE], name="Inputs")
        initial_state = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, STATE_SIZE], name="InitialState")
        initial_hidden_state = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, STATE_SIZE],
                                              name="InitialHiddenState")
        expected_output = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE], name="ExpectedOutputs")
        loss_mask = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, 1], name="lossMaskPlaceholder")

        network_input_split = tf.split(network_input, len(gpus), axis=0)
        initial_state_split = tf.split(initial_state, len(gpus), axis=0)
        initial_hidden_state_split = tf.split(initial_hidden_state, len(gpus), axis=0)
        expected_output_split = tf.split(expected_output, len(gpus), axis=0)
        loss_mask_split = tf.split(loss_mask, len(gpus), axis=0)

    losses = []
    gradients = []
    finalstates = []
    finalhiddenstates = []

    for i in range(0, len(gpus)):
        gpu = gpus[i]
        # This line assures that all following code will be run in the specified device. For a more concrete
        # example of this, see the similar code a few lines down
        with tf.device(gpu):
            loss, finalstate = buildtower(networkinput=network_input_split[i],
                                          initial_state=initial_state_split[i],
                                          initial_hidden_state=initial_hidden_state_split[i],
                                          expected_outputs=expected_output_split[i], loss_mask=loss_mask_split[i])
            gradients.append(optimizer.compute_gradients(loss,
                                                         aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE))
        losses.append(loss)
        finalstates.append(finalstate[0])
        finalhiddenstates.append(finalstate[1])

    # This assures that all the following code will run in the CPU, and not the GPU.
    # More specifically, any graph operation constructed in this "with" block will be run in the CPU
    # when we call session.run(...)
    # (Because remember, this is just graph creation. We aren't actually running anything yet)
    with tf.device("/cpu:0"):
        # This just averages the gradients from each tower
        averagegrads = averagegradients(gradients)

        # And this actually adjusts variables based on the gradient. (Hence why this part needs to be in the CPU)
        train_step = optimizer.apply_gradients(averagegrads)

        # The inputs, hidden states, etc are currently split up between multiple towers.
        # Here, we just stick them all together so we can input them as we did before.
        averagelosses = sum(losses) / len(losses)
        allfinalstates = tf.concat(finalstates, axis=0)
        allfinalhiddenstates = tf.concat(finalhiddenstates, axis=0)

    return network_input, expected_output, initial_state, initial_hidden_state, \
        averagelosses, train_step, loss_mask, allfinalstates, allfinalhiddenstates


def getpaddedbatches(model):
    """
    This function takes the batches of data and shuffles them around to fit
    the format needed by Tensorflow.
    :param model: The language model to use to convert words to vectors
    :return: Maxlength, inputs, outputs, mask, padding.
        Maxlength is the number of timesteps in this batch
        Inputs are the inputs to the network, an ndarray of shape [BATCH_SIZE, maxlength, WORD_VECTOR_SIZE]
        Outputs are the expected outputs from the network, an ndarray of shape [BATCH_SIZE, maxlength, 1]
        Mask is the previously-discussed output mask for padded inputs, an ndarray of shape [BATCH_SIZE, maxlength, 1]
        Padding is a float representing the proportion of this batch that is padding
    """

    for batch in getbatches(BATCH_SIZE):
        # Max length is the number of words in the longest article
        lengths = [len(element[0].split(" ")) for element in batch]
        maxlength = max(lengths)
        padding = 1 - (sum(lengths) / (len(lengths) * maxlength))

        # Inputs are the inputs to the network, in the shape specified by the input placeholder of the graph
        # We default them to zeros, because that is our padding choice
        inputs = np.zeros(shape=(BATCH_SIZE, maxlength, WORD_VECTOR_SIZE), dtype=np.float32)
        # TODO: Deal with different values of WORDS_INPUT_AT_ONCE

        outputs = np.zeros(shape=(BATCH_SIZE, ), dtype=np.float32)

        # We initialize the mask to zeros, then make it 1 whenever we find a word, so that at the end
        # when we run out of words, its all zeros for the rest of the time steps
        mask = np.zeros(shape=(BATCH_SIZE, maxlength, 1), dtype=np.float32)

        for elementNum in range(0, BATCH_SIZE):
            element = batch[elementNum]
            words = element[0].split(" ")
            outputs[elementNum] = (1.0 if element[1] == 'true' else 0.0)
            for i in range(0, len(words)):
                if words[i] != '<UNK>':
                    inputs[elementNum, i] = model[words[i]]
                mask[elementNum, i] = 1.0
        yield maxlength, inputs, outputs, mask, padding


def debugupdate(lossaverage, epochstarttime, batchclusterstarttime):
    """
    This function prints out the average loss and other statistics
    :param lossaverage: Average loss to display
    :param epochstarttime: Time at which this epoch started
    :param batchclusterstarttime: Time at which this group of batches started
    :return:
    """
    now = datetime.datetime.now()
    epochinterval = now - epochstarttime
    batchinterval = now - batchclusterstarttime
    batchseconds = batchinterval.total_seconds()
    print("The last %d batches took %d min %.0f sec total."
          % (UPDATE_FREQUENCY, int(batchseconds / 60), batchseconds % 60))
    print("That's an average of %.1f sec per batch of %d articles." % (batchseconds / UPDATE_FREQUENCY, BATCH_SIZE))
    print("This epoch has so far taken %d min %.1f sec total."
          % (int(epochinterval.total_seconds() / 60), epochinterval.total_seconds() % 60))
    print("The average loss for the epoch so far is %.4f." % lossaverage)


inputs, outputs, initial_state, initial_hidden_state, \
    loss, train_step, loss_mask, finalstate, finalhiddenstate = buildgraph()

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

# This is how we save files to disk
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # The weights and biases for the final output layer are initialized randomly, but the internal
    # weights and biases within the RNN are not. So we need to do this to initialize them.
    try:
        saver.restore(session, VARIABLE_SAVE_FILE)
        print("Restored variables from file %s" % VARIABLE_SAVE_FILE)
    except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
        print("Failed to load variables from file %s." % VARIABLE_SAVE_FILE)
        print("Starting all training over from scratch. Are you sure this is what you want?")
        session.run(tf.global_variables_initializer())

    # One epoch is one run through the entire dataset
    epochNum = 1
    while True:
        print("Starting epoch %d" % epochNum)

        # These two are for averaging the loss over the entire epoch
        lossTotal = 0
        numBatches = 0

        # The time at which the epoch started
        epochstarttime = datetime.datetime.now()

        # The time at which this group of batches started
        batchclusterstarttime = datetime.datetime.now()

        for timeSteps, inputBatch, outputBatch, mask, padding in getpaddedbatches(model):
            numTrue = sum(outputBatch)
            print("Starting batch %d of epoch %d. Max time steps: %d. Proportion padding: %f. True: %d / %d"
                  % (numBatches, epochNum, timeSteps, padding, numTrue, BATCH_SIZE))
            # print(str(mask))

            state = np.zeros(shape=(BATCH_SIZE, STATE_SIZE))
            hidden_state = np.zeros(shape=(BATCH_SIZE, STATE_SIZE))

            # The following 7 lines of code are concerned with finding the best way to split up the time steps
            # of this data for the purposes of truncated backprop. If bad values are chosen, then we can end up
            # with just a couple of words left dangling at the end of the article in their own lonesome batch.
            leftover = timeSteps % TRUNCATION
            if leftover < 0.5 * TRUNCATION:
                numSteps = int(timeSteps / TRUNCATION)
            else:
                numSteps = math.ceil(timeSteps / TRUNCATION)

            stepSize = math.ceil(timeSteps / numSteps)

            # This is where truncated backpropagation comes in. We can't effectively backprop through
            # more than a few hundred time steps, so when articles get too long, we have to break them up
            # into chunks of a few hundred words. This for loop does this.
            # It saves the final state at the end of each chunk, then inputs it as the initial state in
            # the next chunk. This way, the network remembers all the information from the previous chunk,
            # it just can't backpropagate into the previous chunk.
            for i in range(0, timeSteps, stepSize):
                maxTimeStep = i + stepSize

                lossResult, trainStepResult, finalstateresult, finalhiddenstateresult = \
                    session.run([loss, train_step, finalstate, finalhiddenstate], feed_dict={
                        inputs: inputBatch[0: BATCH_SIZE, i: maxTimeStep, 0: WORD_VECTOR_SIZE * WORDS_INPUT_AT_ONCE],
                        outputs: outputBatch,
                        initial_state: state,
                        initial_hidden_state: hidden_state,
                        loss_mask: mask[0: BATCH_SIZE, i: maxTimeStep, 0:1]
                    })

                # Here is where we store the final state from this chunk, to be used by the next chunk
                state = finalstateresult
                hidden_state = finalhiddenstateresult
            # Within one epoch, the loss will bounce around wildly, due to random fluctuations.
            # So it will be more useful to just average the loss over the entire epoch
            print("Finished batch. Loss: %f" % lossResult)
            lossTotal += lossResult
            numBatches += 1

            if numBatches % UPDATE_FREQUENCY == 0:
                print("Saving variables...")
                savepath = saver.save(session, save_path=VARIABLE_SAVE_FILE)
                print("Saved variables to file %s" % savepath)
                debugupdate(lossaverage=(lossTotal / numBatches), epochstarttime=epochstarttime,
                            batchclusterstarttime=batchclusterstarttime)
                batchclusterstarttime = datetime.datetime.now()

        print("Finished epoch %d. Loss average: %f" % (epochNum, (lossTotal / numBatches)))
        epochNum += 1
