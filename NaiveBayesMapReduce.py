from mrjob.job import MRJob
from mrjob.step import MRStep
from collections import Counter


class NaiveBayes(MRJob):

    def mapper(self, _, line):
        data = line.split(',')
        terms, label = data[:-1], data[-1]
        for term in terms:
            yield (term, (label, 1))

    def reducer(self, term, counts):
        label_counts = Counter(dict(counts))
        total_count = sum(label_counts.values())
        yield (term, [(label, count / total_count) for label, count in label_counts.items()])

    def mapper_prior(self, term, label_probs):
        for label, prob in label_probs:
            yield (label, prob)

    def reducer_prior(self, label, probs):
        total_prob = sum(probs)
        yield (label, total_prob)

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer),
            MRStep(mapper=self.mapper_prior, reducer=self.reducer_prior)
        ]


if __name__ == '__main__':
    NaiveBayes.run()
