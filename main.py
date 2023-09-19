import torch
import re
import matplotlib.pyplot as plt

class BigramModel:
    #can chose from words or characters as tokens
    def __init__(self, dataset, mode="words"):
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
        probMatrix = count / sum
        return probMatrix

    def inference(self, num):
        output = []
        for j in range(num):
            generated = ['.']
            i = 0
            while True:
                index = self.token2index[generated[i]]
                tokenIndex = torch.multinomial(self.probMatrix[index], 1, replacement=True).item()
                token = self.index2token[tokenIndex]
                generated.append(token)
                i += 1
                if token == '.':
                    break
            output.append(' '.join(generated))
        return set(output)

if __name__ == '__main__':
    dataset = ('have you eaten .yet', 'i dont think so', 'are you sure', 'actually i am not')
    # names = ('John', 'Jane', 'Jack', 'Janice')
    bigram = BigramModel(dataset)
    # bigram = BigramModel(names, mode="characters")
    print(bigram.inference(10))
