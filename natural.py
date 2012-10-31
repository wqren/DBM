import numpy

import theano
import theano.tensor as T
floatX = theano.config.floatX

def star_prod(s1, s2):
    return (s1.dimshuffle(0,1,'x') * s2.dimshuffle(0,'x',1)).flatten(ndim=2)


def compute_Lx_term1(v, g, h, xw, xv, xa, xb, xc):
    Minv = T.cast(1./v.shape[0], floatX)
    (N0, N1, N2) = (v.shape[1], g.shape[1], h.shape[1])

    vg = star_prod(v,g)
    gh = star_prod(g,h)
    vg_xw = T.dot(vg, xw)
    gh_xv = T.dot(gh, xv)
    v_xa = T.dot(v, xa)
    g_xb = T.dot(g, xb)
    h_xc = T.dot(h, xc)

    def param_rows(lhs_term, rhs_terms):
        param_row = 0.
        for rhs_term in rhs_terms:
            param_row += T.dot(lhs_term, rhs_term)
        return param_row * Minv

    rhs_terms = [vg_xw, gh_xv, v_xa, g_xb, h_xc]
    Lw_rows = param_rows(vg.T, rhs_terms).reshape((N0, N1))
    Lv_rows = param_rows(gh.T, rhs_terms).reshape((N1, N2))
    La_rows = param_rows(v.T, rhs_terms)
    Lb_rows = param_rows(g.T, rhs_terms)
    Lc_rows = param_rows(h.T, rhs_terms)

    return [Lw_rows, Lv_rows, La_rows, Lb_rows, Lc_rows]


def compute_Lx_term2(v, g, h, xw, xv, xa, xb, xc):
    M2inv = T.cast(1./v.shape[0]**2, floatX)
    (N0, N1, N2) = (v.shape[1], g.shape[1], h.shape[1])

    Lw = T.dot(v.T, g).flatten()
    Lv = T.dot(g.T, h).flatten()
    La = T.sum(v, axis=0)
    Lb = T.sum(g, axis=0)
    Lc = T.sum(h, axis=0)

    rhs_term = T.dot(Lw, xw) + T.dot(Lv, xv) +\
               T.dot(La, xa) + T.dot(Lb, xb) + T.dot(Lc, xc)

    rval = [ (Lw * rhs_term).reshape((N0, N1)) * M2inv,
             (Lv * rhs_term).reshape((N1, N2)) * M2inv,
             (La * rhs_term) * M2inv,
             (Lb * rhs_term) * M2inv,
             (Lc * rhs_term)  * M2inv ]

    return rval

def compute_Lx(v, g, h, xw_mat, xv_mat, xa, xb, xc):
    xw = xw_mat.flatten()
    xv = xv_mat.flatten()
    terms1 = compute_Lx_term1(v, g, h, xw, xv, xa, xb, xc)
    terms2 = compute_Lx_term2(v, g, h, xw, xv, xa, xb, xc)
    rval = []
    for (term1, term2) in zip(terms1, terms2):
        rval += [term1 - term2]
    return rval
