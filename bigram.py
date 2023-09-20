import torch
import re
import matplotlib.pyplot as plt

class BigramModel:
    #can chose from words or characters as tokens
    def __init__(self, dataset, mode="characters"):
        self.dataset = dataset
        self.dataset, self.token2index, self.index2token = self.tokenize(self.dataset, mode)
        self.probMatrix = self.count(self.dataset, mode)

    def cleanDataset(self, dataset):
        def containsNonRomanAlphabet(data):
            return not re.match(r'^[a-zA-Z\s]*$', data)
        return [data.lower() for data in dataset if data != '' and not containsNonRomanAlphabet(data)]

    def tokenize(self, dataset, mode):
        dataset = self.cleanDataset(dataset)
        tokens = []
        if mode == "words":
            for data in  dataset:
                tokens.extend(data.split())
        elif mode == "characters":
            for data in dataset:
                tokens.extend(set(data))

        tokens.append('.')
        tokens = sorted(set(tokens))
        token2index = {tokens: i for i, tokens in enumerate(tokens)}
        index2token = {i: tokens for i, tokens in enumerate(token2index)}

        return dataset, token2index, index2token

    def count(self, dataset, mode, smoothing=1):
        bigrams = {}
        for data in dataset:
            data = ('. ' + data + ' .').split() if (mode == "words") else list('.' + data+ '.')
            for token1, token2 in zip(data, data[1:]):
                bigram = (token1, token2)
                bigrams[bigram] = bigrams.get(bigram, 0) + 1

        count = torch.zeros((len(self.token2index), len(self.token2index)), dtype=torch.int32)
        for key in bigrams:
            index1 = self.token2index[key[0]]
            index2 = self.token2index[key[1]]
            count[index1, index2] = bigrams[key]

        sum = torch.sum(count, dim = 1, keepdim=True)
        probMatrix = (count+smoothing) / sum
        print(probMatrix)
        return probMatrix

    def inference(self, num):
        output = []
        negativeLogLikelihoods = []
        for j in range(num):
            generated = ['.']
            i = 0
            negativeLogLikelihood = 0
            while True:
                index = self.token2index[generated[i]]
                tokenIndex = torch.multinomial(self.probMatrix[index], 1, replacement=True).item()
                token = self.index2token[tokenIndex]

                logProb =  torch.log(self.probMatrix[index][tokenIndex])
                # print(generated[i]+token, self.probMatrix[index][tokenIndex], logProb)
                negativeLogLikelihood += logProb
                generated.append(token)

                i += 1
                if token == '.':
                    break

            negativeLogLikelihood = -negativeLogLikelihood/i
            output.append((''.join(generated), negativeLogLikelihood))
        return set(output)

def processFbDataset():
    with open('PH.csv', 'r') as f:
        dataset = f.readlines()
    dataset = [data.strip().split(',') for data in dataset]
    firstNames = [data[0] for data in dataset]
    lastNames = [data[1] for data in dataset]
    return dataset, firstNames, lastNames

if __name__ == '__main__':
    # dataset = ('have you eaten .yet', 'i dont think so', 'are you sure', 'actually i am not')
    # names = ('John', 'Jane', 'Jack', 'Janice')
    _, firstNames, lastNames = processFbDataset()
    bigram = BigramModel(lastNames)
    # bigram = BigramModel(names, mode="characters")

    #output is (generated name, negative log likelihood)
    output = bigram.inference(20)
    for i in output:
        print(i)
