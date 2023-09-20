# Bigram and Trigram Language Model using torch.tensor

This is a script that implements simple Bigram/Trigram Model that generates Filipino names using data from a [Facebook leak in 2021](https://www.businessinsider.com/stolen-data-of-533-million-facebook-users-leaked-online-2021-4).

A bigram/trigram model used to generate sequences of text based on the probability of observing one token (word or character) following 2/3 tokens in a given dataset. I used counting to calculate the probability of the next token (no neural networks here!).

## USAGE

The ?gram.py files generates 20 names by default from the Filipino names dataset.
You can change the main function to generate first names or lastnames, you can also use your own dataset, you can also change the token type to "words" to generate words instead of characters.

## DATA SOURCE

The dataset used in this example is sourced from the "Name Dataset" by Philippe Remy:

- Author: Philippe Remy
- Title: Name Dataset
- Year: 2021
- Publisher: GitHub
- Repository: [Name Dataset GitHub Repository](https://github.com/philipperemy/name-dataset)


<END>
