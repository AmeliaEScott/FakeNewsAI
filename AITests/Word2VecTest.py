"""
This file is for demonstrating the Word2Vec library.
Make sure you've followed the instructions in the README to download
the language model.
"""

from gensim.models import KeyedVectors
import os

dir = os.path.dirname(__file__)
modelpath = os.path.join(dir, "GoogleNews-vectors-negative300.bin")

# This step takes a VERY LONG TIME (1-2 minutes) and LOTS of memory (5 GB), so only do it once in any program!
model = KeyedVectors.load_word2vec_format(modelpath, binary=True)

# This step is also pretty slow, so probably don't use it too often
result = model.most_similar(positive=["cat", "dog"])

print(str(result))

# This is how you turn one word into its vector. (For this model, it's a 300-dimensional vector. AKA, a 300-long array)
vector = model["cat"]
print("Vector for \"cat\":")
print(str(vector))
