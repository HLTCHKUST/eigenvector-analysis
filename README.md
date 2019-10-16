## Interpreting Word Embeddings with Eigenvector Analysis
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="HKUST.jpg" width="12%">

This is the code for:
**Interpreting Word Embeddings with Eigenvector Analysis**. **Jamin Shin**, Andrea Madotto, and Pascale Fung. ***NeurIPS 2018 Workshop on Interpretability and Robustness in Audio, Speech, and Language ([IRASL](https://irasl.gitlab.io/))***. 
[[PDF]](https://openreview.net/forum?id=rJfJiR5ooX)

The code is mainly consisted of Jupyter Notebook and word embedding library [hyperwords](https://bitbucket.org/omerlevy/hyperwords). If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@InProceedings{shin2018interpreting,
  	author = "Shin, Jamin, Madotto, Andrea, and Fung, Pascale",
  	title = 	"Interpreting Word Embeddings with Eigenvector Analysis",
  	booktitle = 	"Workshop on Interpretability and Robustness in Audio, Speech, and Language (IRASL)",
  	year = 	"2018",
  	publisher = "NeurIPS IRASL"
}
</pre>

## Abstract
Dense word vectors have proven their values in many downstream NLP tasks over the past few years. However, the dimensions of such embeddings are not easily interpretable. Out of the d-dimensions in a word vector, we would not be able to understand what high or low values mean. Previous approaches addressing this issue have mainly focused on either training sparse/non-negative constrained word embeddings, or post-processing standard pre-trained word embeddings. On the other hand, we analyze conventional word embeddings trained with Singular Value Decomposition, and reveal similar interpretability. We use a novel eigenvector analysis method inspired from Random Matrix Theory and show that semantically coherent groups not only form in the row space, but also the column space. This allows us to view individual word vector dimensions as human-interpretable semantic features.

## Installation
Our source code is mainly an analysis code using Jupyter Notebook, a modifed version of the Perl-based Wikipedia preprocessing script provided by [Matt Mahoney](http://mattmahoney.net/dc/textdata.html), and [hyperwords](https://bitbucket.org/omerlevy/hyperwords) library by Omer Levy which utilized Python 2.7.

For the data, please download the latest Wikipedia dump, but note that the dump we used which is 2018 April does not exist anymore on [Wikimedia Downloads](https://dumps.wikimedia.org/).

## Training and Testing
All the training and testing followed hyperwords source code, which the documentation is instructed under the [hyperwords.md](hyperwords.md)

## Issues
This code is old and rather unmaintained. However, the main ideas are all shown in the paper and the analysis notebook file. For any potential issues you find, please open a Github Issue.
