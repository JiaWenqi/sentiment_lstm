import six.moves.cPickle as pickle
import numpy as np


def LoadGlove():
  path = '../glove.840B.300d.txt'
  dictionary = {}
  with open(path, 'r') as f:
    count = 0
    while True:
      l = f.readline()
      count += 1
      if count % 10000 == 0:
        print('Read %d lines' % count)
      if not l:
        break
      l = l.lstrip().rstrip()
      word_and_vec = l.split(' ')
      word = word_and_vec[0]
      vec = np.array([float(w) for w in word_and_vec[1:]], dtype=np.float32)
      dictionary[word] = vec
  return dictionary


def LoadImdb():
  path = '../../data/imdb.dict.pkl'
  f = open(path, 'r')
  dictionary = pickle.load(f)
  f.close()
  result = {v: k for k, v in dictionary.items()}
  return result


def main():
  glove_dict = LoadGlove()
  imdb_dict = LoadImdb()

  out_path = '../../data/imdb.glove.emb.pkl'

  emb = np.zeros([len(imdb_dict) + 2, 300], dtype=np.float32)
  for i in range(len(imdb_dict) + 2):
    if i in imdb_dict:
      # not 0 or 1
      word = imdb_dict[i]
      if word in glove_dict:
        # If in glove dict, use the embedding
        emb[i, :] = glove_dict[word]

  pickle.dump(emb, open(out_path, 'wb'))


if __name__ == '__main__':
  main()
