# sentiment_lstm
Implementation of _not exactly_ [this tutorial](http://deeplearning.net/tutorial/lstm.html)

## word embedding
[pre-trained word2vec](https://code.google.com/archive/p/word2vec/)

## dataset
[set 1](https://archive.ics.uci.edu/ml/)

## LSTM tutorial
[A nice tutorial](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## HOW TO
It is not end-to-end complete. You need to download the pretrained embedding for better performance. and create a directory in ../data to store the checkpoint file. But in essence, once all set up, you do either.
1. Train.

   ```python
   python train.py
   ```

2. Predict

   ```python
   python train.py --predict
   ```

###

## TODO
1. ~~Training~~
2. Variable length LSTM. (Currently use 0 padding to the max len)
3. Debug LSTM graph.
4. ~~Import word2vec.~~
5. ~~Reset h after each batch to zero~~
6. ~~Fix exploding loss and nan loss.~~ (This is due to euclidean sqrt function derivative at 0 is nan, end up adding an eps)
7. ~~Should at least use one hot vector to represent word, I doubt it is due to the large value that results in derivative to nan.~~ (Change to embedding)
8. Regularization
9. Dropout
10. padding to the front.
11. ~~checkpointing the model.~~

## Open Question
1. Should we train embedding on the corpse or use pretrained embedding from larger corpse?

## Results
### Index based input without embedding, using the last output, euclidean distance loss
This is the first experiment setting. length is set to around 100, batch_size around 50, the test result never went higher than 55%. This looks miserable result, considering 50% is coin toss.

### Index based input with non-initialized embedding, using the last output, softmax cross entropy loss
learning rate = 5.0
batch_size = 100
length = 150

Achieved ~62% at epoch ~200. At this point, the precision for training set is ~90%, so apparently there is overfitting.

### Index based input with constant pretrained embedding, using the last output, softmax cross entropy loss
learning rate = 5.0
batch_size = 100
length = 150

test precision keeps ~50% before 207 iterations, and *weirdly* suddenly jumps to 98%.
training precision jump at the same time from ~56% to 97%.

I cannot explain why that happened.
