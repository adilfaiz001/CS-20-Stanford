'''
Word2vec
========

At a high level, we need to find an efficient way to represent textual data (in this case, words) so that we can use this
representation to solve natural language tasks. Word embeddings form the backbone in the solutions to many tasks such as 
language modeling, machine translation, sentiment analysis, etc.

Created by a team of researchers led by Tomas Mikolov, word2vec is a group of models that are used to produce word embeddings. 
There are two main models used in word2vec: skip-gram and CBOW. 


Skip-gram vs CBOW (Continuous Bag-of-Words)
-------------------------------------------
Algorithmically, these models are similar, except that CBOW predicts center words from context words, while the skip-gram does 
the inverse and predicts source context-words from the center words. For example, if we have the sentence: ""The quick brown fox 
jumps"", then CBOW tries to predict ""brown"" from ""the"", ""quick"", ""fox"", and ""jumps"", while skip-gram tries to predict 
""the"", ""quick"", ""fox"", and ""jumps"" from ""brown"".

Statistically it has the effect that CBOW smoothes over a lot of the distributional information (by treating an entire context 
as one observation). For the most part, this turns out to be a useful thing for smaller datasets. However, skip-gram treats each 
context-target pair as a new observation, and this tends to do better when we have larger datasets.

Softmax, Negative Sampling, and Noise Contrastive Estimation
------------------------------------------------------------
To get the distribution of the possible neighboring words, in theory, we often use softmax. Softmax maps arbitrary values xi to a 
probability distribution pi. In this case, softmax(xi) is the probability that xi is a neighboring word of a specific word we are 
considering.

softmax(xi) = exp(xi) / iexp(xi)

However, the normalization term in the denominator requires us to perform exp on all words in the dictionary and sum the results 
up, which could be millions of words. Even if you disregard uncommon words, a natural language model doesn’t perform well unless 
you consider at least tens of thousands of the most common words. The normalization term causes softmax to be computationally 
prohibitive.

There are two main approaches to circumvent this bottleneck: hierarchical softmax and sample-based softmax.

Negative sampling, as the name suggests, belongs to the family of sample-based approaches. This family also includes importance 
sampling and target sampling. Negative sampling is actually a simplified model of an approach called Noise Contrastive Estimation 
(NCE), e.g. negative sampling makes certain assumption about the number of noise samples to generate -- let’s call it k -- and 
the distribution of noise samples -- let’s call it Q -- such that kQ(w) = 1 to simplify computation.












t-SNE (from Wikipedia)
-----------------------
t-distributed stochastic neighbor embedding (t-SNE) is a machine learning algorithm for dimensionality reduction developed by 
Geoffrey Hinton and Laurens van der Maaten. It is a nonlinear dimensionality reduction technique that is particularly well-
suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter 
plot. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar 
objects are modeled by nearby points and dissimilar objects are modeled by distant points. 

The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional 
objects in such a way that similar objects have a high probability of being picked, whilst dissimilar points have an extremely 
small probability of being picked. Second, t-SNE defines a similar probability distribution over the points in the low-
dimensional map, and it minimizes the Kullback–Leibler divergence between the two distributions with respect to the locations of 
the points in the map. Note that whilst the original algorithm uses the Euclidean distance between objects as the base of its 
similarity metric, this should be changed as appropriate.

'''