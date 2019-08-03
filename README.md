# Sequence Encoding OUTDATED 
## Implemented two word representations
* ### TFIDF_Representation (done -to be implemented with input from file or tokenization)
  
  + <a href="https://www.codecogs.com/eqnedit.php?latex=$TermFrequency&space;=&space;f_{t,d}&space;/&space;\sum_{t'&space;\in&space;d}&space;f_{t',d}$\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$TermFrequency&space;=&space;f_{t,d}&space;/&space;\sum_{t'&space;\in&space;d}&space;f_{t',d}$\\" title="$TermFrequency = f_{t,d} / \sum_{t' \in d} f_{t',d}\t$ " /></a>
  t = term, d = document , f(t,d) = raw count\\
  + <a href="https://www.codecogs.com/eqnedit.php?latex=$Inverse_Document_Frequency&space;=&space;\log{\cfrac{N}{n_t}}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$InverseDocumentFrequency&space;=&space;\log{\cfrac{N}{n_t}}$" title="$Inverse_Document_Frequency = \log{\cfrac{N}{n_t}}\t$" /></a>
  N = total number of documents
* ### Contribution of word to probabilistic topics (done with demo input from file- to check tokenizations)
  + Probabilistically estimate X topics
  + for each term calculate X contributions to topics
  + represent each term as a vector of contributions, <c1,c2,...cX>

# Neural Network Classifier
## LSTM RNN 
### Supervised learning - input dataset of sentences with labels (keep/not to keep)
+ Evaluate the two representations
+ Compare results between languages
