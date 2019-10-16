from embedding import SVDEmbedding, EnsembleEmbedding, Embedding
from explicit import PositiveExplicit


def create_representation(args):
    rep_type = args['<representation>']
    path = args['<representation_path>']
    neg = int(args['--neg'])
    w_c = args['--w+c']
    eig = float(args['--eig'])
    normalize = args['--normalize']

    if rep_type == 'PPMI':
        if w_c:
            raise Exception('w+c is not implemented for PPMI.')
        else:
            return PositiveExplicit(path, normalize, neg)

    elif rep_type == 'SVD':
        if w_c:
            return EnsembleEmbedding(SVDEmbedding(path, normalize, eig, False), SVDEmbedding(path, normalize, eig, True), normalize)
        else:
            return SVDEmbedding(path, normalize, eig)

    else:
        if w_c:
            return EnsembleEmbedding(Embedding(path + '.words', normalize), Embedding(path + '.contexts', normalize), normalize)
        else:
            return Embedding(path + '.words', normalize)
