# Bigram and Trigram Language Model using torch.tensor

This is a script that implements simple Bigram/Trigram Model that generates Filipino names using (horrible) data from a [Facebook leak in 2021](https://www.businessinsider.com/stolen-data-of-533-million-facebook-users-leaked-online-2021-4).

## Understanding the models 

The Bigram and Trigram models are classes that take a dataset and a mode ("words" or "characters") as inputs. The dataset is tokenized based on the chosen mode, and a probability matrix is created by counting the occurrences of each token following a pair (Bigram) or a trio (Trigram) of tokens.

The models generate new sequences of text using the inference method, which generates a specified number of names. The generation process uses the probability matrix to select the next token based on the previous tokens.

The output of the inference method is a set of generated names. For the Bigram model, each name is accompanied by its negative log likelihood (NLL). The NLL is a measure of the model's confidence in the generated sequence: the lower the NLL, the higher the model's confidence.

## USAGE

The bigram.py and trigram.py scripts generate 20 names by default from a Filipino names dataset. You can modify the main function in these scripts to generate different numbers of names, use different datasets, or change the token type to "words" to generate sequences of words instead of characters.

Here's an example of how to use the Bigram model to generate 10 names:

```
bigram = BigramModel(dataset)
output = bigram.inference(10)
```
## DATA SOURCE

The dataset used in this example is sourced from the "Name Dataset" by Philippe Remy, published on GitHub in 2021. You can find the dataset [here](https://github.com/philipperemy/name-dataset).
