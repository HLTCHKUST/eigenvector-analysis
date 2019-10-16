from __future__ import print_function

from collections import Counter
from math import sqrt
from random import Random, uniform

from docopt import docopt

import pickle


def main():
    args = docopt("""
    Usage:
        corpus2pairs.py [options] <corpus>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
        --pos        Positional contexts
        --dyn        Dynamic context windows
        --sub NUM    Subsampling threshold [default: 0]
        --del        Delete out-of-vocabulary and subsampled placeholders
        --loadv      Load vocab of threshold
        --loads      Load subsampled vocab of threshold
        --prob NUM   random sampling drop probability of articles
    """)

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])
    pos = args['--pos']
    dyn = args['--dyn']
    subsample = float(args['--sub'])
    sub = subsample != 0
    d3l = args['--del']
    load_vocab = args['--loadv']
    load_sub = args['--loads']
    prob = float(args['--prob'])

    vocab = read_vocab(corpus_file, thr, load_vocab)
    corpus_size = sum(vocab.values())

    if not load_vocab:
        pickle.dump(vocab, open('./vocab_{}.pkl'.format(thr), 'wb'))
        # print(len(vocab))
        # print(corpus_size)

    if sub:
        if load_sub:
            subsampler = pickle.load(
                open("sub_vocab_{}.pkl".format(thr), "rb"))
        else:
            # print(subsample)
            subsample *= corpus_size
            # print(subsample)
            subsampler = dict([(word, 1 - sqrt(subsample / count))
                               for word, count in vocab.items() if count > subsample])
            pickle.dump(subsampler, open(
                './sub_vocab_{}.pkl'.format(thr), 'wb'))
            # print(len(subsampler))

    rnd = Random(17)
    with open(corpus_file) as f:
        for line in f:
            if prob:
                if uniform(0, 1) >= prob:
                    continue

            tokens = [t if t in vocab else None for t in line.strip().split()]
            if sub:
                tokens = [t if t not in subsampler or rnd.random() > subsampler[
                    t] else None for t in tokens]
            if d3l:
                tokens = [t for t in tokens if t is not None]

            len_tokens = len(tokens)

            for i, tok in enumerate(tokens):
                if tok is not None:
                    if dyn:
                        dynamic_window = rnd.randint(1, win)
                    else:
                        dynamic_window = win
                    start = i - dynamic_window
                    if start < 0:
                        start = 0
                    end = i + dynamic_window + 1
                    if end > len_tokens:
                        end = len_tokens

                    if pos:
                        output = '\n'.join([row for row in [tok + ' ' + tokens[j] + '_' + str(j - i) for j in xrange(
                            start, end) if j != i and tokens[j] is not None] if len(row) > 0]).strip()
                    else:
                        output = '\n'.join([row for row in [tok + ' ' + tokens[j] for j in xrange(
                            start, end) if j != i and tokens[j] is not None] if len(row) > 0]).strip()
                    if len(output) > 0:
                        print(output)


def read_vocab(corpus_file, thr, load):
    if load:
        return pickle.load(open("vocab_{}.pkl".format(thr), "rb"))
    else:
        vocab = Counter()

        with open(corpus_file) as f:
            for line in f:
                for w in line.strip().split():
                    vocab[w] += 1
        # print(len(vocab))
        corpus_size = sum(vocab.values())
        # print(corpus_size)
        return dict([(token, count) for token, count in vocab.items() if count >= thr])


if __name__ == '__main__':
    main()
