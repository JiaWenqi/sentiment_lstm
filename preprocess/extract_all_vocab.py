'''Load the imdb dictionary(word: index), and output each word as a line into a text file.'''

import six.moves.cPickle as pickle


def main():
  path = '../../data/imdb.dict.pkl'
  out_path = '../../data/word_list.txt'
  data = pickle.load(open(path))

  with open(out_path, 'w') as f:
    for k in data.keys():
      f.write('%s\n' % k)


if __name__ == '__main__':
  main()
