import psycopg2
import os
import json
import tensorflow as tf

# Size of batch in number of articles
# Batch size of 1 means each article is its own batch
BATCH_SIZE = 5

# Size of a vector for an individual word
WORD_VECTOR_SIZE = 300

# Number of words to input to the network at a time
WORDS_INPUT_AT_ONCE = 1

# Size of state to remember between iterations within one article
STATE_SIZE = 3000


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
networkinput = tf.placeholder(tf.float32, [BATCH_SIZE, None, WORD_VECTOR_SIZE * WORDS_INPUT_AT_ONCE])

# The LSTM model has both a state and a hidden state (don't ask me what the difference is). Whenever the
# network is run, it uses the state from the previous time it was run. For our use, each time we give
# the network a word, it uses the state from the previous word. However, for the first word, it has no
# context, so we need to give it one. By using a placeholder instead of a constant, we can choose to
# either just give it all zeroes, or maybe carry over the context from another article, if that ever
# becomes useful (it won't).
#
# This state is how the network remembers context as it works its way through an article. Because it takes more
# than one word's worth of information to remember the context of an entire article, our state needs to carry way
# more than 300 values. In our case, we chose 3000. (This value may change, and I might forget to update this comment,
# so keep that in mind.)
# The first number, BATCH_SIZE, means the same thing as it did before. TensorFlow processes an
# entire batch in parallel, so it needs to know how big a batch is.
# The second number, STATE_SIZE, is what was just discussed: In our usage, this number effectively represents
# how much information the network remembers about the article.
initial_state = tf.placeholder(tf.float32, [BATCH_SIZE, STATE_SIZE])

# This is basically the same as the initial_state above, but for the hidden state. I don't know what
# the difference is between the state and the hidden state.
initial_hidden_state = tf.placeholder(tf.float32, [BATCH_SIZE, STATE_SIZE])

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
# The first, outputs, is the outputs from the network (SHOCKER). This is a tensor that contains every
# output (in our case, STATE_SIZE of them), for every time step (AKA, every word in the article), for every input
# in the batch. Thus, it is a rank-3 tensor (read: 3D matrix) of size [BATCH_SIZE, TIME_STEPS, STATE_SIZE].
# Remember that second dimension, TIME_STEPS, which was "None" before? Well hopefully, TensorFlow is smart enough
# to fill that in with an actual number by the time we need it.
# TODO: I don't actually know if this is true. I should find out
# The second value, state, is simply a tensor representing the final state after the network has run
# through all of its inputs. It is a rank-2 Tensor (Read: 2D matrix) of size [BATCH_SIZE, STATE_SIZE].
# Remember how TensorFlow processes everything in a batch at once? Well, this state needs to have a value
# for everything in the batch, thus why the output is this size.
outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=networkinput, initial_state=initial_state_tuple)





