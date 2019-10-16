from sparsesvd import sparsesvd
# from gensim.models import LsiModel
import scipy.sparse.linalg as la
import dask.array as da

from docopt import docopt
import numpy as np

from representations.explicit import PositiveExplicit
from representations.matrix_serializer import save_vocabulary


def main():
    args = docopt("""
    Usage:
        pmi2svd.py [options] <pmi_path> <output_path>
    
    Options:
        --dim NUM    Dimensionality of eigenvectors [default: 500]
        --neg NUM    Number of negative samples; subtracts its log from PMI [default: 1]
        --sig NUM    Shift invert mode [default: -1]
        --mod NUM    0 for LM, 1 for SM, 2 for BE [default: 0]
    """)

    pmi_path = args['<pmi_path>']
    output_path = args['<output_path>']
    dim = int(args['--dim'])
    neg = int(args['--neg'])
    sig = float(args['--sig'])
    sig = sig if sig >= 0 else None
    mode = int(args['--mod'])
    if mode == 1:
        mode = 'SM'
    elif mode == 2:
        mode = 'BE'
    else:
        mode = 'LM'

    explicit = PositiveExplicit(pmi_path, normalize=False, neg=neg)

    # ut, s, vt = sparsesvd(explicit.m.tocsc(), dim)
    # s, u = la.eigsh(explicit.m.tocsc(), k=dim, sigma=sig,
    #                 which=mode, return_eigenvectors=True)
    print explicit.m.shape
    ppmi = da.from_array(explicit.m, chunks=(427, 61))
    print ppmi.ndim
    print len(ppmi.chunks[1])
    u, s, vt = da.linalg.svd_compressed(ppmi, dim * 2, 4)
    print s

    s = s.compute()
    # u, s, vt = la.svds(explicit.m.tocsc(), k=dim,
    #                    which=mode, return_singular_vectors=True)

    # s, u = la.eigs(explicit.m.tocsc(), k=dim,
    #                which='SR', return_eigenvectors=True)

    np.save(output_path + '.s.npy', s)

    u = u.compute()
    np.save(output_path + '.u.npy', u)
    # np.save(output_path + '.vt.npy', vt)
    save_vocabulary(output_path + '.words.vocab', explicit.iw)
    save_vocabulary(output_path + '.contexts.vocab', explicit.ic)


if __name__ == '__main__':
    main()
