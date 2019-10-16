from __builtin__ import sorted

from docopt import docopt
import numpy as np

from representations.representation_factory import create_representation


def main():
    args = docopt("""
    Usage:
        analogy_eval.py [options] <representation> <representation_path> <task_path>
    
    Options:
        --neg NUM    Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c        Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM    Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
        --normalize  Use row-normalized word vectors
    """)

    print args['--normalize']
    representation = create_representation(args)
    args['--normalize'] = True
    print args['--normalize']
    representation_sim = create_representation(args)
    data = read_test_set(representation, args['<task_path>'])
    xi, ix = get_vocab(data)
    accuracy_add, accuracy_mul = evaluate(
        representation, representation_sim, data, xi, ix)
    print args['<representation>'], args['<representation_path>'], '\t%0.3f' % accuracy_add, '\t%0.3f' % accuracy_mul


def read_test_set(representation, path):
    test = []
    unks = 0
    with open(path) as f:
        for line in f:
            analogy = line.strip().lower().split()
            a, b, c, d = analogy
            if a not in representation.wi or b not in representation.wi or c not in representation.wi or d not in representation.wi:
                unks += 1
                continue
            test.append(analogy)

    print 'skipped ' + str(unks) + ' unk sets'

    return test


def get_vocab(data):
    vocab = set()
    for analogy in data:
        vocab.update(analogy)
    vocab = sorted(vocab)
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab


def evaluate(representation, representation_sim, data, xi, ix):
    sims = prepare_similarities(representation_sim, ix)
    correct_add = 0.0
    correct_mul = 0.0
    for a, a_, b, b_ in data:
        b_add, b_mul = guess(representation, sims, xi, a, a_, b)
        if b_add == b_:
            correct_add += 1
        if b_mul == b_:
            correct_mul += 1
    return correct_add / len(data), correct_mul / len(data)


def prepare_similarities(representation, vocab):
    vocab_representation = representation.m[
        [representation.wi[w] for w in vocab]]
    # vocab_representation = representation.m[
    #     [representation.wi[w] if w in representation.wi else 0 for w in vocab]]
    sims = vocab_representation.dot(representation.m.T)

    # dummy = None
    # for w in vocab:
    #     if w not in representation.wi:
    #         dummy = representation.represent(w)
    #         break
    # if dummy is not None:
    #     for i, w in enumerate(vocab):
    #         if w not in representation.wi:
    #             vocab_representation[i] = dummy

    if type(sims) is not np.ndarray:
        sims = np.array(sims.todense())
    else:
        sims = (sims + 1) / 2
    return sims


def guess(representation, sims, xi, a, a_, b):
    sa = sims[xi[a]]
    sa_ = sims[xi[a_]]
    sb = sims[xi[b]]

    add_sim = -sa + sa_ + sb
    if a in representation.wi:
        add_sim[representation.wi[a]] = 0
    if a_ in representation.wi:
        add_sim[representation.wi[a_]] = 0
    if b in representation.wi:
        add_sim[representation.wi[b]] = 0
    b_add = representation.iw[np.nanargmax(add_sim)]

    mul_sim = sa_ * sb * np.reciprocal(sa + 0.01)
    if a in representation.wi:
        mul_sim[representation.wi[a]] = 0
    if a_ in representation.wi:
        mul_sim[representation.wi[a_]] = 0
    if b in representation.wi:
        mul_sim[representation.wi[b]] = 0
    b_mul = representation.iw[np.nanargmax(mul_sim)]

    return b_add, b_mul


if __name__ == '__main__':
    main()
