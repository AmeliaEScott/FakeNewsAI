# AI Tests

These scripts are used for testing various AI libraries. Setup might be a bit cumbersome, so pay attention.

### Python libraries

Each of these libraries is used by at least one of these scripts, so it's probably not
a bad idea to just install them all using pip.

 - Tensorflow
   - I had trouble installing Tensorflow using pip, so I had to do it this way:
   - `pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.0-py3-none-any.whl`
 - gensim
 - psycopg2
 
### Data

To use the word2vec library, you need to download the language model 
from [here](https://docs.google.com/file/d/0B7XkCwpI5KDYaDBDQm1tZGNDRHc/edit)

Put it right here in the AITests folder. Make sure to unzip it, and name it exactly `GoogleNews-vectors-negative300.bin`.
(This is important because these scripts will have to reference this file by name)

Do **not** commit this file to the repo. It's 1.6 gigabytes.