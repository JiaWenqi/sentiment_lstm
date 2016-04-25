# sentiment_lstm
Implementation of _not exactly_ [this tutorial](http://deeplearning.net/tutorial/lstm.html)

## word embedding
[pre-trained word2vec](https://code.google.com/archive/p/word2vec/)

## dataset
[set 1](https://archive.ics.uci.edu/ml/)

## Reference
* [A nice tutorial](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [Visualize LSTM](http://arxiv.org/pdf/1506.02078v2.pdf)
* [Data-driven document](https://d3js.org/)
* [Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml)
* [CS224D report](https://cs224d.stanford.edu/reports/HongJames.pdf)
* [New book of Deep learning from Goodfellow etc.](http://www.deeplearningbook.org/contents/regularization.html)

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

3. Analyze

   ```python
   python train.py --analyze
   python -m SimpleHTTPServer
   ```

   Go to [http://localhost:8000/test.html](http://localhost:8000/test.html) to view activation of each cell after reading each word.

## TODO
1. ~~Training~~
2. Variable length LSTM. (Currently use 0 padding to the max len)
3. Debug LSTM graph.
4. ~~Import word2vec.~~
5. ~~Reset h after each batch to zero~~
6. ~~Fix exploding loss and nan loss.~~ (This is due to euclidean sqrt function derivative at 0 is nan, end up adding an eps)
7. ~~Should at least use one hot vector to represent word, I doubt it is due to the large value that results in derivative to nan.~~ (Change to embedding)
8. ~~Regularization~~ L2. If the weight is high, e.g. 0.1 then it doesn't train at all.
9. ~~Dropout~~. Helps with overfitting quite a bit.
10. ~~padding to the front.~~ It seems to train faster.
11. ~~checkpointing the model.~~
12. ~~Clip gradient.~~
13. Weight decay.
14. Bucketing.
15. ~~manual Hyper param tuning.~~ It seems that state_size = 20 is good, but I haven't tried larger. learning_rate seems to be good below 0.01 for state_size = 20
16. Early stop
17. ~~Use get_variable.~~ Concatenate 4 gate weight into one as well.
18. ~~Understand initialization.~~ Initialize using math.sqrt(fan_in) for weights to sharpen the activation. Orthognal initialization doesn't help the fluctuation between mini batches.
19. ~~units for C.~~ Add a logistic regression layer for the last output.
20. autotuning hyper param.
21. GloVe vector.
22. ~~AdaOptmizer~~. AdaGrad greatly reduce the fluctuation.
23. ~~Color code the words for cell activation.~~
24. interactive web service to do sentiment analysis.
25. ~~Preprocess special characters such as '~~. Use the same tokenizer as that for embedding.

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

This result is not reproducible. In fact, it might be related to the initialization in a long stride. A few more runs shows the precision to peak around 65% and then have a sudden drop in precision and increase in cost.

### Index based input with constant pretrained embedding, using the last output, softmax cross entropy loss, variable state size, add a last logistic regression layer to convert from h to logits.
Keeping the following constant,
length = 50
batch_size = 5
state_size = 10

Tuning learning rate,
learning_rate = 0.1 seems to be the fastest yet stable rate. Achieving 99.59% accuracy.

However, there seems to be serious overfitting. The best validation accuracy achieved is 60%

Keeping the following constant,
length = 50
batch_size = 5
learning_rate = 0.1

Tuning state_size,
state_size = 10 is the fastest yet stable size.

At state_size = 15, the overfitting seems better, achieving at best 80% validation accuracy.

At state_size = 20, the overfitting seems even better, achieving a closer gap most of the time. However, at this state_size, learning_rate = 0.01 results in too larg step at around 80% training accuracy.

Train on larger corpse, after about 15 hours,
length = 150
batch_size = 100
learning_rate = 0.001
state_size = 30
training accuracy increase to 99.92%, validation accuracy to 73.17%, test accuracy to 73.66%

**Accuracy for different mini batch varies a lot.**

### replace SGD with AdaGrad
clipping by +/-5.0
After ~20 iterations, accuracy rose to 85%. The curve for accuracy looks asymptotic with reasonable volatility.

### Trace activation after each word update for each cell.
Need to visualize them.

### Replace word2vec with Glove vector(840B)
with trivial preprocessing,
70% of word has vector in Glove, while only 45% of word has vector in word2vec
There is a boost in accuracy and learning speed using glove.

length = 100
batch_size = 100
At epoch = 30
Glove give ~95% training accuracy and 85% validation accuracy
word2vec gives ~90% training accuracy and similar validation accuracy.

### Preprocess using Stanford parser
Standford parse and one give [here](https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl) differs on how *hasn't*, *won't*, etc is tokenized, *'t* is not in glove vector.
Tokenizing results in many tokens that are not in the embedding, but should make sense to human, such as, *director/writer/main* *late-spring/summer*

The accuracy after 30 epochs is similar to using the other parser, that is ~95% for training, and 85% for validation

### Increase learning rate and loosen clipping.
learning_rate = 0.05 (from 0.005)
clip = +/-1000.0 (from +/-5.0, in real case, much gradients are capped by +/- 5.0, and are generally < +/- 80.0), so this has the same effect as no clipping.
length = 100
batch_size = 100

After 17 iteration, training precision is 99.9%, validation precision is 89%. Model overfit.
Aparrently, Adagrad doesn't require small clipping and learning_rate, and provide stability at the same time.

### Dropout to input
Apply drop out to the input. Dropout set weight at w/p for input neurons that are not disabled. dropout = 0.5, hidden_unit = 150
dropout = 0.5, hidden_unit = 100 => validation accuracy = 94%
dropout = 1.0, hidden_unit = 50 => validation accuracy ~90%

Dropout helps reduce the overfit quite a bit.
