import numpy

import theano
import theano.tensor as T
import theano.sandbox.scan
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
             (Lc * rhs_term) * M2inv ]

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

def compute_Lx_batches(v, g, h, xw_mat, xv_mat, xa, xb, xc, bs, cbs):
    xw = xw_mat.flatten()
    xv = xv_mat.flatten()
    tv = v.reshape((bs // cbs, cbs, v.shape[1]))
    tg = g.reshape((bs // cbs, cbs, g.shape[1]))
    th = h.reshape((bs // cbs, cbs, h.shape[1]))

    final_w1 = T.unbroadcast(T.shape_padleft(T.zeros_like(xw_mat)),0)
    final_v1 = T.unbroadcast(T.shape_padleft(T.zeros_like(xv_mat)),0)
    final_a1 = T.unbroadcast(T.shape_padleft(T.zeros_like(xa)),0)
    final_b1 = T.unbroadcast(T.shape_padleft(T.zeros_like(xb)),0)
    final_c1 = T.unbroadcast(T.shape_padleft(T.zeros_like(xc)),0)
    def comp_step(lv, lg, lh,
                  acc_w1, acc_v1, acc_a1, acc_b1, acc_c1):
        terms1 = compute_Lx_term1(lv, lg, lh, xw, xv, xa, xb, xc)
        accs1 = [acc_w1, acc_v1, acc_a1, acc_b1, acc_c1]
        rval = []

        for (term1, acc) in zip(terms1,accs1):
            rval += [acc + term1]
        return rval
    rvals,_ = theano.sandbox.scan.scan(
        comp_step,
        sequences=[tv,tg,th],
        states=[
            final_w1, final_v1, final_a1, final_b1, final_c1],
        n_steps=bs // cbs,
        profile=0,
        mode=theano.Mode(linker='cvm_nogc'),
        flags=['no_optimization'] )
    accs1 = [x[0]/numpy.float32(bs//cbs) for x in rvals]
    accs2 = compute_Lx_term2(v,g,h,xw,xv,xa,xb,xc)
    return [x - y for x, y in zip(accs1, accs2)]

def compute_L_diag(v, g, h):
    Minv = T.cast(1./v.shape[0], floatX)
    (M, N0, N1, N2) = (v.shape[0], v.shape[1], g.shape[1], h.shape[1])

    dE_dW = T.dot(v.T, g).flatten() * Minv
    dE_dV = T.dot(g.T, h).flatten() * Minv
    Lww = T.mean(star_prod(v,g)**2, axis=0) - dE_dW**2
    Lvv = T.mean(star_prod(g,h)**2, axis=0) - dE_dV**2
    Laa = T.mean(v**2, axis=0) - T.mean(v, axis=0)**2
    Lbb = T.mean(g**2, axis=0) - T.mean(g, axis=0)**2
    Lcc = T.mean(h**2, axis=0) - T.mean(h, axis=0)**2

    return [Lww.reshape((N0,N1)), Lvv.reshape((N1,N2)), Laa, Lbb, Lcc]
