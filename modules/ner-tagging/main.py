import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque


def log_sum_exp_stable(values, dim):
    alpha_values = torch.max(values, dim = dim).values

    return alpha_values + torch.log(torch.exp(values.T - alpha_values).sum(dim = 0).clamp(min = 1e-8))


class CRF:

    LEARNING_RATE = 0.1

    weights: torch.Tensor

    def __init__(self, labels, verbose = False):
        self.verbose = verbose

        self.labels, self.labels_size = labels, len(labels)
        self.weights = nn.Parameter(torch.randn(self.labels_size, self.labels_size))

        self.optimizer = optim.SGD([self.weights], lr = CRF.LEARNING_RATE)

    def forward(self, emissions, targets):
        assert(emissions.size(1) == self.labels_size)

        # Clear residual gradients
        self.optimizer.zero_grad()

        # Compute the total output & the best seq output
        # T size vec
        prev, target_prob = emissions[0, :], emissions[0][targets[0]]
        for i in range(1, emissions.size(0)):
            # T x T row wise
            prev = prev.repeat(self.labels_size, 1).T + emissions[i].repeat(self.labels_size, 1).T + self.weights
            prev = log_sum_exp_stable(prev, dim = 1)

            target_prob = target_prob + emissions[i][targets[i]] + self.weights[targets[i - 1]][targets[i]]

        total_prob = log_sum_exp_stable(prev, dim = 0)

        # Compute loss and accumulate gradients
        loss = total_prob - target_prob
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.weights, 0.5)

        if self.verbose:
            print(f"{total_prob} -> {target_prob}\tLoss -> {loss.item()}")

        # Optimize step
        self.optimizer.step()

    def decode(self, emissions) -> torch.Tensor:
        # Initialize
        seq_size, targets = emissions.size(0), []

        # Memoization of max prob for each t timestamp and indexing for backtracking
        T1, T2 = torch.zeros((seq_size, self.labels_size)), torch.zeros((seq_size, self.labels_size))

        # Initialize starting labels/targets with emission output
        T1[0] = emissions[0]

        for seq_index in range(1, seq_size):
            for prev_index in range(self.labels_size):
                for index in range(self.labels_size):
                    transition_end_score = T1[seq_index - 1][prev_index] + self.weights[prev_index, index] + \
                                           emissions[seq_index][index]

                    if transition_end_score > T1[seq_index][index]:
                        T1[seq_index][index], T2[seq_index][index] = transition_end_score, prev_index

        # Multinomial sampling
        index = torch.argmax(T1[-1])

        decoding = deque([index])
        for t in range(seq_size - 1, 0, -1):
            index = int(T2[t][index])

            decoding.appendleft(index)

        return torch.LongTensor(decoding)


if __name__ == '__main__':
    epochs, seq_size = 300, 16

    torch.autograd.set_detect_anomaly(True)

    # Our World
    labels = ['a', 'b', 'c', 'd', 'e']

    # Gold Labels
    targets = torch.zeros((seq_size,)).long()

    # Emission scores with small gaussian noise
    emissions = torch.randn(len(labels),).repeat(seq_size, 1) + torch.distributions.normal.Normal(0, 0.1).sample((seq_size, len(labels)))

    crf = CRF(labels, verbose = True)
    for i in range(epochs):
        crf.forward(emissions, targets)

    print(f"\tTrue Targets {targets} -> {crf.decode(emissions)}")