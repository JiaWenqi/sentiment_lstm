'''Read a text file of word and embedding and pickle dump a numpy array, where each row is embedding for that index.'''

import six.moves.cPickle as pickle
import numpy as np


def main():
  embedding_size = 300
  path = '../../data/imdb.dict.pkl'
  dictionary = pickle.load(open(path))
  out_path = '../../data/imdb.emb.pkl'

  path = '../../data/output.txt'
  # Account for missing index 0 and 1.
  emb = np.zeros([len(dictionary) + 2, embedding_size], dtype=np.float32)
  with open(path, 'r') as f:
    while True:
      word = f.readline()
      if not word:
        break
      word = word.rstrip()
      embeddings = f.readline().rstrip().split()
      embeddings = [float(e) for e in embeddings]
      if word in dictionary:
        emb[int(dictionary[word]), :] = np.array(embeddings, dtype=np.float32)

  pickle.dump(emb, open(out_path, 'wb'))


if __name__ == '__main__':
  main()
