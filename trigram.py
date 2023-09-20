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
        def removePunctuation(sentence):
            return re.sub(r'[^\w\s]', '', sentence)
        return [removePunctuation(sentence) for sentence in dataset]

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
        tokens = set(tokens)
        token2index = {tokens: i for i, tokens in enumerate(tokens)}
        index2token = {i: tokens for i, tokens in enumerate(token2index)}

        return dataset, token2index, index2token

    def count(self, dataset, mode):
        trigrams = {}
        for data in dataset:
            data = ('. . ' + data + ' . .').split() if (mode == "words") else list('..' + data+ '..')
            for token1, token2, token3 in zip(data, data[1:], data[2:]):
                trigram = (token1, token2, token3)
                trigrams[trigram] = trigrams.get(trigram, 0) + 1

        count = torch.zeros((len(self.token2index), len(self.token2index), len(self.token2index)), dtype=torch.int32)
        for key in trigrams:
            index1 = self.token2index[key[0]]
            index2 = self.token2index[key[1]]
            index3 = self.token2index[key[2]]
            count[index1, index2, index3] = trigrams[key]

        sum = torch.sum(count, dim = 2, keepdim=True)
        probMatrix = count / sum
        return probMatrix

    def inference(self, num):
        output = []
        for j in range(num):
            generated = ['.', '.']
            i = 1
            while True:
                index1 = self.token2index[generated[i-1]]
                index2 = self.token2index[generated[i]]
                tokenIndex = torch.multinomial(self.probMatrix[index1][index2], 1, replacement=True).item()
                token = self.index2token[tokenIndex]
                generated.append(token)
                i += 1
                if token == '.':
                    break
            output.append(''.join(generated))
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
    output = bigram.inference(20)
    for i in output:
        print(i[2:-1])
