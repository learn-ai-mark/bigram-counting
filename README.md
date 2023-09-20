# Bigram Language Model using torch.tensor

This is a script that implements simple Bigram Model that generates Filipino names using data from a [Facebook leak in 2021](https://www.businessinsider.com/stolen-data-of-533-million-facebook-users-leaked-online-2021-4).

A bigram model used to generate sequences of text based on the probability of observing one token (word or character) following another token in a given dataset. I used counting to calculate the probability of the next token (no neural networks here!).

## USAGE

The main.py file generates 20 names by default from the Filipino names dataset.

## DATA SOURCE

The dataset used in this example is sourced from the "Name Dataset" by Philippe Remy:

- Author: Philippe Remy
- Title: Name Dataset
- Year: 2021
- Publisher: GitHub
- Repository: [Name Dataset GitHub Repository](https://github.com/philipperemy/name-dataset)


<END>
