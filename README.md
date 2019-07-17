#Sequence Encoding
Implemented two word representations
* TFIDF_representation (ongoing)
+ <a href="https://www.codecogs.com/eqnedit.php?latex=$TermFrequency&space;=&space;f_{t,d}&space;/&space;\sum_{t'&space;\in&space;d}&space;f_{t',d}$\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$TermFrequency&space;=&space;f_{t,d}&space;/&space;\sum_{t'&space;\in&space;d}&space;f_{t',d}$\\" title="$TermFrequency = f_{t,d} / \sum_{t' \in d} f_{t',d}$\\" /></a>
  t = term,
d = document
f(t,d) = raw count
+ <a href="https://www.codecogs.com/eqnedit.php?latex=$Inverse_Document_Frequency&space;=&space;\log{\cfrac{N}{n_t}}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$InverseDocumentFrequency&space;=&space;\log{\cfrac{N}{n_t}}$" title="$Inverse_Document_Frequency = \log{\cfrac{N}{n_t}}$" /></a>
  N = total number of documents
* Contribution of word to probabilistic topics (ongoing)
