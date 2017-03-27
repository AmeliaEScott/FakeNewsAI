"""
This was our first pass at training the network.
It works to some extent, but it does not utilize multiple GPUs.
We are keeping it around because it serves as our personal Tensorflow reference
"""

import sys

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    print("You forgot to install Tensorflow and Numpy.")
    sys.exit()

import os

try:
    from gensim.models import KeyedVectors
except ImportError:
    print("You forgot to install gensim. Run pip install gensim")
    sys.exit()

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
BATCH_SIZE = 100

# Size of a vector for an individual word
WORD_VECTOR_SIZE = 300

# Number of words to input to the network at a time
WORDS_INPUT_AT_ONCE = 1

# Size of state to remember between iterations within one article
STATE_SIZE = 3000

# Number of GPUs on the target machine. Can be 0
NUM_GPUS = 8

# The learning rate is a constant for how quickly it should learn.
# Too high, and it could overshoot and oscillate around wildly. Too low,
# and it'll just take forever. I have no idea how to find the right value.
LEARNING_RATE = 1.0


def buildgraph():
    """
    This function builds the computational graph.
    In TensorFlow terminology, the "graph" is a pre-built set of computations, that will be actually executed
    later using a session. This function builds a graph for a (very specific) sequence-to-sequence RNN.
    :return: A computational graph.
    """
    # This represents the inputs to the network.
    # The first number is the batch size: Because TensorFlow will process an entire batch in parallel,
    # it needs to know how many inputs there are in a batch. Each input in a batch is processes independently,
    # so for us, a batch will represent several articles, and each article will be processed independently.
    # The next number is the number of "time steps", or the number of times to run through the network.
    # In our case, that is the number of words in an article. However, because every article has a different
    # number of words, we can't put an actual number here. Luckily, TensorFlow lets us choose "None", which
    # tells TensorFlow to just fill in the right number for that every time the graph is run.
    # The last number is the number of actual inputs to the actual network. For us, each word is a 300-long
    # vector (array), so each word needs 300 inputs. If we input multiple words at a time, then the number
    # of inputs is the number of words input at once times 300.
    networkinput = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, WORD_VECTOR_SIZE * WORDS_INPUT_AT_ONCE],
                                  name="inputPlaceholder")

    # The LSTM model has both a state and a hidden state (don't ask me what the difference is). Whenever the
    # network is run, it uses the state from the previous time it was run. For our use, each time we give
    # the network a word, it uses the state from the previous word. However, for the first word, it has no
    # context, so we need to give it one. By using a placeholder instead of a constant, we can choose to
    # either just give it all zeroes, or maybe carry over the context from another article, if that ever
    # becomes useful (it won't).
    #
    # This state is how the network remembers context as it works its way through an article. Because it takes more
    # than one word's worth of information to remember the context of an entire article, our state needs to carry way
    # more than 300 values. In our case, we chose 3000. (This value may change, and I might forget
    # to update this comment, so keep that in mind.)
    # The first number, BATCH_SIZE, means the same thing as it did before. TensorFlow processes an
    # entire batch in parallel, so it needs to know how big a batch is.
    # The second number, STATE_SIZE, is what was just discussed: In our usage, this number effectively represents
    # how much information the network remembers about the article.
    initial_state = tf.placeholder(tf.float32, shape=[BATCH_SIZE, STATE_SIZE], name="initialStatePlaceholder")

    # This is basically the same as the initial_state above, but for the hidden state. I don't know what
    # the difference is between the state and the hidden state.
    initial_hidden_state = tf.placeholder(tf.float32, shape=[BATCH_SIZE, STATE_SIZE],
                                          name="initialHiddenStatePlaceholder")

    # This just sticks those two states together, because this is what the TensorFlow Gods demand.
    initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(initial_state, initial_hidden_state)

    # A recurrent network has many methods of passing its state back to itself. This cell determines which
    # of those methods to use. In our case, we're using Long Short-Term Memory, or LSTM. For reasons.
    cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)

    # HERE is where the actual RNN is made. It should be noted that this RNN can be used as a component of a larger,
    # more complex network. The way this function works, the number of output nodes can't be chosen. In this case, it
    # is equal to the state size of the cell, in our case, STATE_SIZE (Maybe 3000, unless we change it).
    # To solve this issue, we'll feed this output into a few more layers of "normal" neural network to
    # reduce it down to the single output we want. TODO: Actually try this, see if it works.
    #
    # Here's how this function works: Cell is a type of cell. The cell is described above. We're specifically
    # using a LSTM cell.
    # Inputs is a placeholder that represents the actual inputs to the network. It has to have a time
    # dimension, as described above. (The time dimension is the one that is "None")
    # Initial_state is the state that this network starts out with. For us, it is the palceholders described above.
    # If cell is a LSTM cell, then initial_state has to be a tf.contrib.rnn.LSTMStateTuple (for reasons).
    #
    # This function returns two values (which is something Python can do).
    # The first, rnn_outputs, is the outputs from the network (SHOCKER). This is a tensor that contains every
    # output (in our case, STATE_SIZE of them), for every time step (AKA, every word in the article), for every input
    # in the batch. Thus, it is a rank-3 tensor (read: 3D matrix) of size [BATCH_SIZE, TIME_STEPS, STATE_SIZE].
    # Remember that second dimension, TIME_STEPS, which was "None" before? Well hopefully, TensorFlow is smart enough
    # to fill that in with an actual number by the time we need it.
    # TODO: I don't actually know if this is true. I should find out
    # The second value, state, is simply a tensor representing the final state after the network has run
    # through all of its inputs. It is a rank-2 Tensor (Read: 2D matrix) of size [BATCH_SIZE, STATE_SIZE].
    # Remember how TensorFlow processes everything in a batch at once? Well, this state needs to have a value
    # for everything in the batch, thus why the output is this size.
    rnn_outputs, finalstate = tf.nn.dynamic_rnn(cell=cell, inputs=networkinput, initial_state=initial_state_tuple)

    # The function get_shape() could be helpful for debugging
    # print(rnn_outputs.get_shape())

    # Variables differ from placeholders because they can be trained by Tensorflow's training functions.
    # Weights and Biases are, conveniently, two things that need to be trained, so we use variables for those.
    # The following weights and biases are for the connections between the output of the recurrent neural network
    # (Which is STATE_SIZE long), and our final output (which is only 1 long), hence the shape of [STATE_SIZE, 1]
    weights = tf.Variable(initial_value=np.random.rand(STATE_SIZE, 1), dtype=tf.float32,
                          expected_shape=[STATE_SIZE, 1])

    # While weights are on the connections between neurons, biases are on the neurons themselves.
    # This particular bias is on our final output neuron, which is why the shape is just [1]. It's literally
    # just a single value.
    biases = tf.Variable(initial_value=np.zeros(1), dtype=tf.float32, expected_shape=[1])

    # The rnn_outputs are currently a jumble of way too many dimensions. (More specifically,
    # its shape is [BATCH_SIZE, time_steps (None), STATE_SIZE]). In order to simplify training, we want to
    # simply "unpack" it so it's simply one long series of outputs. So in the end, it should have the
    # shape of [something, STATE_SIZE]. The tf.reshape() function does this exactly. It just takes all of the
    # values in the tensor and unfolds them to the specified shape. The -1 is just to tell Tensorflow to
    # calculate that value as needed.
    rnn_outputs_reshaped = tf.reshape(rnn_outputs, [-1, STATE_SIZE])

    # Here is where the "normal" neural network layer is. It just connects the outputs of the neural network
    # to our final, single cell. Sigmoid is a function that just constrains the output to between 0 and 1.
    # That plus sign doesn't actually add any numbers together, because remember, we're not doing any computations
    # yet, we're just building the computational graph. The plus sign just tells tensorflow to add a "plus"
    # node to the graph.
    # (Operator overloading in Python is a fun thing)
    network_outputs = tf.sigmoid(tf.matmul(rnn_outputs_reshaped, weights) + biases)

    # So now, after all of this, we have a few things set up:
    #  - A few placeholders for inputting the input values and the initial state(s)
    #  - A recurrent neural network that iterates over those inputs to create a series of outputs
    #  - A final layer to reduce the series of rnn states down to a series of single outputs (network_outputs)
    # That, in theory, would be enough to run the network if it had already been trained. But to train it,
    # we need a few more things:
    #  - Placeholders for the expected outputs for the given inputs
    #  - The actual training steps
    # We'll start with those expected outputs.
    # Because the expected outputs change with every batch (just like the inputs), we'll use
    # placeholders (just like with the inputs). The BATCH_SIZE in the shape is there for the same reason
    # as it is in every other placeholder. The None is for the number of time steps, just like before. The 1
    # is because there is only one actual output from our network (the "real" or "fake" estimate).
    expected_outputs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, 1], name="expectedOutputsPlaceholder")

    # Just like with the rnn_outputs, we have to reshape it to be easier to work with.
    # We could have just constructed the initial placeholder to be this shape, but this way makes more sense to me.
    expected_outputs_reshaped = tf.reshape(expected_outputs, [-1, 1])

    # Remember from earlier how every thing in a batch has to be the same length?
    # Well, we can't reasonably do that, so we have to pad the shorter articles with zeroes.
    # However, we can't expect the network to give us a good answer for those zeroes, since there
    # is no good answer. So, we don't let the losses (errors) from these zeroes contribute to our overall
    # loss. To do this, we multiply every output by 0 if it's padded, so that it's completely ignored.
    # This is the same shape as the outputs, because it has to be element-wise multiplied by the outputs.
    loss_mask = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, 1], name="lossMaskPlaceholder")

    # Reshape it for the same reason we reshape the outputs
    loss_mask_reshaped = tf.reshape(loss_mask, shape=[-1])

    # Here is where we actually multiply the mask times the outputs.
    # (There's probably a better way to do this, but this will probably work)
    expected_outputs_reshaped = loss_mask_reshaped * expected_outputs_reshaped
    network_outputs = loss_mask_reshaped * network_outputs

    # Loss is another word for error, or how wrong our network is. This is necessary for training.
    # "labels" are the correct answers, that the network should get.
    # "predictions" are the actual answers, that the network actually got.
    # Mean squared error is super simple and just finds the sum of the squares of the differences between these.
    # The result is just a single number, or, more accurately, a Tensor of shape [1].
    loss = tf.losses.mean_squared_error(labels=expected_outputs_reshaped, predictions=network_outputs)

    # This is the step that actually does the training.
    # It updates any and all variables in the graph to minimize whatever tensor is given by loss.
    # I'm sure loss could be pretty much anything you want, but in this case, we want to minimize
    # errors because that just makes sense.
    train_step = tf.train.AdagradOptimizer(0.3).minimize(loss)

    return networkinput, expected_outputs, initial_state, initial_hidden_state, loss, train_step, loss_mask


def getpaddedbatches(model):
    """
    This function takes the batches of data and shuffles them around to fit
    the format needed by Tensorflow.
    :param model: The language model to use to convert words to vectors
    :return: Maxlength, inputs, outputs, mask.
        Maxlength is the number of timesteps in this batch
        Inputs are the inputs to the network
        Outputs are the expected outputs from the network
        Mask is the previously-discussed output mask for padded inputs
    """

    for batch in getbatches(BATCH_SIZE):
        # Max length is the number of words in the longest article
        maxlength = max([len(element[0].split(" ")) for element in batch])

        # Inputs are the inputs to the network, in the shape specified by the input placeholder of the graph
        # We default them to zeros, because that is our padding choice
        inputs = np.zeros(shape=(BATCH_SIZE, maxlength, WORD_VECTOR_SIZE), dtype=np.float32)
        # TODO: Deal with different values of WORDS_INPUT_AT_ONCE

        outputs = np.zeros(shape=(BATCH_SIZE, maxlength, 1), dtype=np.float32)

        # We initialize the mask to zeros, then make it 1 whenever we find a word, so that at the end
        # when we run out of words, its all zeros for the rest of the time steps
        mask = np.zeros(shape=(BATCH_SIZE, maxlength, 1), dtype=np.float32)

        for elementNum in range(0, BATCH_SIZE):
            element = batch[elementNum]
            words = element[0].split(" ")
            output = 1.0 if element[1] == 'true' else 0.0
            for i in range(0, len(words)):
                if words[i] != '<UNK>':
                    inputs[elementNum, i] = model[words[i]]
                outputs[elementNum, i] = output
                mask[elementNum, i] = 1.0
        yield maxlength, inputs, outputs, mask


inputs, outputs, initial_state, initial_hidden_state, loss, train_step, loss_mask = buildgraph()

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

with tf.Session() as session:
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

        # These two are for averaging the loss over the epoch
        lossTotal = 0
        numBatches = 0

        for timeSteps, inputBatch, outputBatch, mask in getpaddedbatches(model):
            # feed_dict is how we pass in values for all the placeholders
            # This is the part of this code that takes all of the time and processor power.
            # Even though the batching code is stupidly inefficient, it takes negligible time compared
            # to the actual training, so let's not bother optimizing it.
            lossResult, trainStepResult = session.run([loss, train_step], feed_dict={
                inputs: inputBatch,
                outputs: outputBatch,
                initial_state: np.zeros((BATCH_SIZE, STATE_SIZE)),
                initial_hidden_state: np.zeros((BATCH_SIZE, STATE_SIZE)),
                loss_mask: mask
            })
            # Within one epoch, the loss will bounce around wildly, due to random fluctuations.
            # So it will be more useful to just average the loss over the entire epoch
            print("Loss: %f" % lossResult)
            lossTotal += lossResult
            numBatches += 1

            if numBatches % 100 == 0:
                print("Saving variables...")
                savepath = saver.save(session, save_path=VARIABLE_SAVE_FILE)
                print("Saved variables to file %s" % savepath)

        print("Finished epoch %d. Loss average: %f" % (epochNum, (lossTotal / numBatches)))
        epochNum += 1
