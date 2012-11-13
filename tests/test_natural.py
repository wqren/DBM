import numpy
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

from DBM import natural
from DBM import minres
from DBM import lincg

rng = numpy.random.RandomState(92832)
(M,N0,N1,N2) = (256,3,4,5)
v = rng.randint(low=0, high=2, size=(M,N0)).astype('float32')
g = rng.randint(low=0, high=2, size=(M,N1)).astype('float32')
h = rng.randint(low=0, high=2, size=(M,N2)).astype('float32')
xw_mat = (numpy.dot(v.T, g) / M).astype('float32')
xv_mat = (numpy.dot(g.T, h) / M).astype('float32')
xw = xw_mat.flatten()
xv = xv_mat.flatten()
xa = numpy.mean(v, axis=0)
xb = numpy.mean(g, axis=0)
xc = numpy.mean(h, axis=0)
x = numpy.hstack((xw, xv, xa, xb, xc))

vg = (v[:,:,None] * g[:,None,:]).reshape(M,-1)
gh = (g[:,:,None] * h[:,None,:]).reshape(M,-1)

# W-rows
Lww = numpy.dot(vg.T, vg)
Lwv = numpy.dot(vg.T, gh)
Lwa = numpy.dot(vg.T, v)
Lwb = numpy.dot(vg.T, g)
Lwc = numpy.dot(vg.T, h)
# V-rows
Lvw = numpy.dot(gh.T, vg)
Lvv = numpy.dot(gh.T, gh)
Lva = numpy.dot(gh.T, v)
Lvb = numpy.dot(gh.T, g)
Lvc = numpy.dot(gh.T, h)
# a-rows
Law = numpy.dot(v.T, vg)
Lav = numpy.dot(v.T, gh)
Laa = numpy.dot(v.T, v)
Lab = numpy.dot(v.T, g)
Lac = numpy.dot(v.T, h)
# b-rows
Lbw = numpy.dot(g.T, vg)
Lbv = numpy.dot(g.T, gh)
Lba = numpy.dot(g.T, v)
Lbb = numpy.dot(g.T, g)
Lbc = numpy.dot(g.T, h)
# c-rows
Lcw = numpy.dot(h.T, vg)
Lcv = numpy.dot(h.T, gh)
Lca = numpy.dot(h.T, v)
Lcb = numpy.dot(h.T, g)
Lcc = numpy.dot(h.T, h)

Lw = numpy.dot(v.T, g).flatten()
Lv = numpy.dot(g.T, h).flatten()
# mean will be taken later
La = numpy.sum(v, axis=0)
Lb = numpy.sum(g, axis=0)
Lc = numpy.sum(h, axis=0)

Minv = numpy.float32(1./M)
M2inv = numpy.float32(1./M**2)

L1 = numpy.vstack((
        numpy.hstack((Lww, Lwv, Lwa, Lwb, Lwc)),
        numpy.hstack((Lvw, Lvv, Lva, Lvb, Lvc)),
        numpy.hstack((Law, Lav, Laa, Lab, Lac)),
        numpy.hstack((Lbw, Lbv, Lba, Lbb, Lbc)),
        numpy.hstack((Lcw, Lcv, Lca, Lcb, Lcc)),
        )) * Minv

L2 = numpy.outer(
            numpy.hstack((Lw, Lv, La, Lb, Lc)),
            numpy.hstack((Lw, Lv, La, Lb, Lc))) * M2inv

L = L1 - L2
Lx = numpy.dot(L, x)
Lx_w = Lx[:N0*N1].reshape(N0,N1)
Lx_v = Lx[N0*N1 : N0*N1 + N1*N2].reshape(N1,N2)
Lx_a = Lx[N0*N1 + N1*N2 : N0*N1 + N1*N2 + N0]
Lx_b = Lx[N0*N1 + N1*N2 + N0 : N0*N1 + N1*N2 + N0 + N1]
Lx_c = Lx[-N2:]

## now compute L^-1 x
Linv_x = linalg.cho_solve(linalg.cho_factor(L), x)
Linv_x_w = Linv_x[:N0*N1].reshape(N0,N1)
Linv_x_v = Linv_x[N0*N1 : N0*N1 + N1*N2].reshape(N1,N2)
Linv_x_a = Linv_x[N0*N1 + N1*N2 : N0*N1 + N1*N2 + N0]
Linv_x_b = Linv_x[N0*N1 + N1*N2 + N0 : N0*N1 + N1*N2 + N0 + N1]
Linv_x_c = Linv_x[-N2:]


def test_compute_Lx_batches():

    ## now compare against theano version
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    aa = T.vector()
    bb = T.vector()
    cc = T.vector()
    xxw_mat = T.matrix()
    xxv_mat = T.matrix()
    xxw = T.vector()
    xxv = T.vector()
    xxa = T.vector()
    xxb = T.vector()
    xxc = T.vector()

    # test compute_Lx
    LLx = natural.compute_Lx_batches(vv, gg, hh, xxw_mat, xxv_mat, xxa, xxb, xxc,
                             256, 64)
    f = theano.function([vv, gg, hh, xxw_mat, xxv_mat, xxa, xxb, xxc], LLx)
    rvals = f(v, g, h, xw_mat, xv_mat, xa, xb, xc)
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[4], decimal=3)

def test_compute_Lx():

    ## now compare against theano version
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    aa = T.vector()
    bb = T.vector()
    cc = T.vector()
    xxw_mat = T.matrix()
    xxv_mat = T.matrix()
    xxw = T.vector()
    xxv = T.vector()
    xxa = T.vector()
    xxb = T.vector()
    xxc = T.vector()

    # test compute_Lx
    LLx = natural.compute_Lx(vv, gg, hh, xxw_mat, xxv_mat, xxa, xxb, xxc)
    f = theano.function([vv, gg, hh, xxw_mat, xxv_mat, xxa, xxb, xxc], LLx)
    rvals = f(v, g, h, xw_mat, xv_mat, xa, xb, xc)
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[4], decimal=3)

def test_generic_compute_Lx_batches():

    ## now compare against theano version
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    aa = T.vector()
    bb = T.vector()
    cc = T.vector()
    xxw_mat = T.matrix()
    xxv_mat = T.matrix()
    xxw = T.vector()
    xxv = T.vector()
    xxa = T.vector()
    xxb = T.vector()
    xxc = T.vector()

    # test compute_Lx
    LLx = natural.generic_compute_Lx_batches([vv, gg, hh],
                                             [xxw_mat, xxv_mat],
                                             [xxa, xxb, xxc],
                                             256, 64)
    f = theano.function([vv, gg, hh, xxw_mat, xxv_mat, xxa, xxb, xxc], LLx)
    rvals = f(v, g, h, xw_mat, xv_mat, xa, xb, xc)
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[4], decimal=3)

def test_generic_compute_Lx():

    ## now compare against theano version
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    aa = T.vector()
    bb = T.vector()
    cc = T.vector()
    xxw_mat = T.matrix()
    xxv_mat = T.matrix()
    xxw = T.vector()
    xxv = T.vector()
    xxa = T.vector()
    xxb = T.vector()
    xxc = T.vector()

    # test compute_Lx
    LLx = natural.generic_compute_Lx([vv, gg, hh],
                                     [xxw_mat, xxv_mat],
                                     [xxa, xxb, xxc])
    f = theano.function([vv, gg, hh, xxw_mat, xxv_mat, xxa, xxb, xxc], LLx)
    rvals = f(v, g, h, xw_mat, xv_mat, xa, xb, xc)
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[4], decimal=3)

def test_math():
    """
    A hand-unrolled implementation of the math in 2012/dbm_natural/math.lyx.
    We compare against the L matrix which is computed more efficiently above,
    using dot and Khatri-Rao products. L is then compared to our theano
    implementation of L in test_compute_Lx.
    """
    # second, dummy implementation of L
    N = N0*N1 + N1*N2 + N0 + N1 + N2

    EL = numpy.zeros((N,N))

    idx = {}
    idx.update({'ws': 0, 'we': N0*N1})
    idx.update({'vs': idx['we'], 've': idx['we'] + N1*N2})
    idx.update({'as': idx['ve'], 'ae': idx['ve'] + N0})
    idx.update({'bs': idx['ae'], 'be': idx['ae'] + N1})
    idx.update({'cs': idx['be'], 'ce': idx['be'] + N2})

    dW = numpy.dot(v.T, g) / M
    dV = numpy.dot(g.T, h) / M
    da = numpy.mean(v, axis=0)
    db = numpy.mean(g, axis=0)
    dc = numpy.mean(h, axis=0)

    # WW block
    for i in xrange(N0):
        for j in xrange(N1):
            r = i*N1 + j
            for k in xrange(N0):
                for l in xrange(N1):
                    c = k*N1 + l
                    EL[r, c] = 0
                    for z in xrange(M):
                        EL[r, c] += v[z,i] * g[z,j] * v[z,k] * g[z,l] - dW[i,j] * dW[k,l]
                    EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['ws']:idx['we'], idx['ws']:idx['we']],
                                      EL[idx['ws']:idx['we'], idx['ws']:idx['we']],
                                      decimal=3)

    # WV block
    for i in xrange(N0):
        for j in xrange(N1):
            r = i*N1 + j
            for k in xrange(N1):
                for l in xrange(N2):
                    c = N0*N1 + k*N2 + l
                    EL[r, c] = 0
                    for z in xrange(M):
                        EL[r, c] += v[z,i] * g[z,j] * g[z,k] * h[z,l] - dW[i,j] * dV[k,l]
                    EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['ws']:idx['we'], idx['vs']:idx['ve']],
                                      EL[idx['ws']:idx['we'], idx['vs']:idx['ve']],
                                      decimal=3)

    # Wa block
    for i in xrange(N0):
        for j in xrange(N1):
            r = i*N1 + j
            for k in xrange(N0):
                c = N0*N1 + N1*N2 + k
                EL[r, c] = 0
                for z in xrange(M):
                    EL[r, c] += v[z,i] * g[z,j] * v[z,k] - dW[i,j] * da[k]
                EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['ws']:idx['we'], idx['as']:idx['ae']],
                                      EL[idx['ws']:idx['we'], idx['as']:idx['ae']],
                                      decimal=3)

    # Wb block
    for i in xrange(N0):
        for j in xrange(N1):
            r = i*N1 + j
            for k in xrange(N1):
                c = N0*N1 + N1*N2 + N0 + k
                EL[r, c] = 0
                for z in xrange(M):
                    EL[r, c] += v[z,i] * g[z,j] * g[z,k] - dW[i,j] * db[k]
                EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['ws']:idx['we'], idx['bs']:idx['be']],
                                      EL[idx['ws']:idx['we'], idx['bs']:idx['be']],
                                      decimal=3)
    # Wc block
    for i in xrange(N0):
        for j in xrange(N1):
            r = i*N1 + j
            for k in xrange(N2):
                c = N0*N1 + N1*N2 + N0 + N1 + k
                EL[r, c] = 0
                for z in xrange(M):
                    EL[r, c] += v[z,i] * g[z,j] * h[z,k] - dW[i,j] * dc[k]
                EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['ws']:idx['we'], idx['cs']:idx['ce']],
                                      EL[idx['ws']:idx['we'], idx['cs']:idx['ce']],
                                      decimal=3)
    # VV block
    for i in xrange(N1):
        for j in xrange(N2):
            r = N0*N1 + i*N2 + j
            for k in xrange(N1):
                for l in xrange(N2):
                    c = N0*N1 + k*N2 + l
                    EL[r, c] = 0
                    for z in xrange(M):
                        EL[r, c] += g[z,i] * h[z,j] * g[z,k] * h[z,l] - dV[i,j] * dV[k,l]
                    EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['vs']:idx['ve'], idx['vs']:idx['ve']],
                                      EL[idx['vs']:idx['ve'], idx['vs']:idx['ve']],
                                      decimal=3)
    # Va block
    for i in xrange(N1):
        for j in xrange(N2):
            r = N0*N1 + i*N2 + j
            for k in xrange(N0):
                c = N0*N1 + N1*N2 + k
                EL[r, c] = 0
                for z in xrange(M):
                    EL[r, c] += g[z,i] * h[z,j] * v[z,k] - dV[i,j] * da[k]
                EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['vs']:idx['ve'], idx['as']:idx['ae']],
                                      EL[idx['vs']:idx['ve'], idx['as']:idx['ae']],
                                      decimal=3)
    # Vb block
    for i in xrange(N1):
        for j in xrange(N2):
            r = N0*N1 + i*N2 + j
            for k in xrange(N1):
                c = N0*N1 + N1*N2 + N0 + k
                EL[r, c] = 0
                for z in xrange(M):
                    EL[r, c] += g[z,i] * h[z,j] * g[z,k] - dV[i,j] * db[k]
                EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['vs']:idx['ve'], idx['bs']:idx['be']],
                                      EL[idx['vs']:idx['ve'], idx['bs']:idx['be']],
                                      decimal=3)
    # Vc block
    for i in xrange(N1):
        for j in xrange(N2):
            r = N0*N1 + i*N2 + j
            for k in xrange(N2):
                c = N0*N1 + N1*N2 + N0 + N1 + k
                EL[r, c] = 0
                for z in xrange(M):
                    EL[r, c] += g[z,i] * h[z,j] * h[z,k] - dV[i,j] * dc[k]
                EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['vs']:idx['ve'], idx['cs']:idx['ce']],
                                      EL[idx['vs']:idx['ve'], idx['cs']:idx['ce']],
                                      decimal=3)
    # aa block
    for i in xrange(N0):
        r = N0*N1 + N1*N2 + i
        for k in xrange(N0):
            c = N0*N1 + N1*N2 + k
            EL[r, c] = 0
            for z in xrange(M):
                EL[r, c] += v[z,i] * v[z,k] - da[i] * da[k]
            EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['as']:idx['ae'], idx['as']:idx['ae']],
                                      EL[idx['as']:idx['ae'], idx['as']:idx['ae']],
                                      decimal=3)
    # ab block
    for i in xrange(N0):
        r = N0*N1 + N1*N2 + i
        for k in xrange(N1):
            c = N0*N1 + N1*N2 + N0 + k
            EL[r, c] = 0
            for z in xrange(M):
                EL[r, c] += v[z,i] * g[z,k] - da[i] * db[k]
            EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['as']:idx['ae'], idx['bs']:idx['be']],
                                      EL[idx['as']:idx['ae'], idx['bs']:idx['be']],
                                      decimal=3)
    # ac block
    for i in xrange(N0):
        r = N0*N1 + N1*N2 + i
        for k in xrange(N2):
            c = N0*N1 + N1*N2 + N0 + N1 + k
            EL[r, c] = 0
            for z in xrange(M):
                EL[r, c] += v[z,i] * h[z,k] - da[i] * dc[k]
            EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['as']:idx['ae'], idx['cs']:idx['ce']],
                                      EL[idx['as']:idx['ae'], idx['cs']:idx['ce']],
                                      decimal=3)
    # bb block
    for i in xrange(N1):
        r = N0*N1 + N1*N2 + N0 + i
        for k in xrange(N1):
            c = N0*N1 + N1*N2 + N0 + k
            EL[r, c] = 0
            for z in xrange(M):
                EL[r, c] += g[z,i] * g[z,k] - db[i] * db[k]
            EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['bs']:idx['be'], idx['bs']:idx['be']],
                                      EL[idx['bs']:idx['be'], idx['bs']:idx['be']],
                                      decimal=3)
    # bc block
    for i in xrange(N1):
        r = N0*N1 + N1*N2 + N0 + i
        for k in xrange(N2):
            c = N0*N1 + N1*N2 + N0 + N1 + k
            EL[r, c] = 0
            for z in xrange(M):
                EL[r, c] += g[z,i] * h[z,k] - db[i] * dc[k]
            EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['bs']:idx['be'], idx['cs']:idx['ce']],
                                      EL[idx['bs']:idx['be'], idx['cs']:idx['ce']],
                                      decimal=3)
    # cc block
    for i in xrange(N2):
        r = N0*N1 + N1*N2 + N0 + N1 + i
        for k in xrange(N2):
            c = N0*N1 + N1*N2 + N0 + N1 + k
            EL[r, c] = 0
            for z in xrange(M):
                EL[r, c] += h[z,i] * h[z,k] - dc[i] * dc[k]
            EL[r,c] /= M
    numpy.testing.assert_almost_equal( L[idx['cs']:idx['ce'], idx['cs']:idx['ce']],
                                      EL[idx['cs']:idx['ce'], idx['cs']:idx['ce']],
                                      decimal=3)

    # make matrix symmetric
    EL[idx['vs']:idx['ve'], idx['ws']:idx['we']] = EL[idx['ws']:idx['we'], idx['vs']:idx['ve']].T
    EL[idx['as']:idx['ae'], idx['ws']:idx['we']] = EL[idx['ws']:idx['we'], idx['as']:idx['ae']].T
    EL[idx['as']:idx['ae'], idx['vs']:idx['ve']] = EL[idx['vs']:idx['ve'], idx['as']:idx['ae']].T
    EL[idx['bs']:idx['be'], idx['ws']:idx['we']] = EL[idx['ws']:idx['we'], idx['bs']:idx['be']].T
    EL[idx['bs']:idx['be'], idx['vs']:idx['ve']] = EL[idx['vs']:idx['ve'], idx['bs']:idx['be']].T
    EL[idx['bs']:idx['be'], idx['as']:idx['ae']] = EL[idx['as']:idx['ae'], idx['bs']:idx['be']].T
    EL[idx['cs']:idx['ce'], idx['ws']:idx['we']] = EL[idx['ws']:idx['we'], idx['cs']:idx['ce']].T
    EL[idx['cs']:idx['ce'], idx['vs']:idx['ve']] = EL[idx['vs']:idx['ve'], idx['cs']:idx['ce']].T
    EL[idx['cs']:idx['ce'], idx['as']:idx['ae']] = EL[idx['as']:idx['ae'], idx['cs']:idx['ce']].T
    EL[idx['cs']:idx['ce'], idx['bs']:idx['be']] = EL[idx['bs']:idx['be'], idx['cs']:idx['ce']].T
    numpy.testing.assert_almost_equal(L, EL, decimal=3)

def test_compute_Ldiag():

    ## now compare against theano version
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    # test compute_Lx
    LL = natural.compute_L_diag(vv, gg, hh)
    f = theano.function([vv, gg, hh], LL)
    rvals = f(v, g, h)
    # compare against baseline
    Ldiag = numpy.diag(L)
    Ldiag_w = Ldiag[:N0*N1]
    Ldiag_v = Ldiag[N0*N1 : N0*N1 + N1*N2]
    Ldiag_a = Ldiag[-N2-N1-N0:-N2-N1]
    Ldiag_b = Ldiag[-N2-N1:-N2]
    Ldiag_c = Ldiag[-N2:]
    numpy.testing.assert_almost_equal(Ldiag_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Ldiag_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Ldiag_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Ldiag_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Ldiag_c, rvals[4], decimal=3)


def test_minres():
    vv = theano.shared(v, name='v')
    gg = theano.shared(g, name='g')
    hh = theano.shared(h, name='h')
    dw = T.dot(v.T,g) / M
    dv = T.dot(g.T,h) / M
    da = T.mean(v, axis=0)
    db = T.mean(g, axis=0)
    dc = T.mean(h, axis=0)
   
    newgrads = minres.minres(
            lambda xw, xv, xa, xb, xc: natural.compute_Lx(vv,gg,hh,xw,xv,xa,xb,xc),
            [dw, dv, da, db, dc],
            rtol=1e-5,
            damp = 0.,
            maxit = 30,
            profile=0)[0]

    f = theano.function([], newgrads)
    [new_dw, new_dv, new_da, new_db, new_dc] = f()
    numpy.testing.assert_almost_equal(Linv_x_w, new_dw, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_v, new_dv, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_a, new_da, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_b, new_db, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_c, new_dc, decimal=1)


def test_minres_with_xinit():
    rng = numpy.random.RandomState(123412)

    vv = theano.shared(v, name='v')
    gg = theano.shared(g, name='g')
    hh = theano.shared(h, name='h')
    dw = T.dot(v.T,g) / M
    dv = T.dot(g.T,h) / M
    da = T.mean(v, axis=0)
    db = T.mean(g, axis=0)
    dc = T.mean(h, axis=0)
  
    xinit = [ rng.rand(N0,N1),
              rng.rand(N1,N2),
              rng.rand(N0),
              rng.rand(N1),
              rng.rand(N2)]
    xinit = [xi.astype('float32') for xi in xinit]

    newgrads = minres.minres(
            lambda xw, xv, xa, xb, xc: natural.compute_Lx(vv,gg,hh,xw,xv,xa,xb,xc),
            [dw, dv, da, db, dc],
            rtol=1e-5,
            damp = 0.,
            maxit = 30,
            xinit = xinit,
            profile=0)[0]

    f = theano.function([], newgrads)
    [new_dw, new_dv, new_da, new_db, new_dc] = f()
    numpy.testing.assert_almost_equal(Linv_x_w, new_dw, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_v, new_dv, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_a, new_da, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_b, new_db, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_c, new_dc, decimal=1)


def test_linearcg():
    vv = theano.shared(v, name='v')
    gg = theano.shared(g, name='g')
    hh = theano.shared(h, name='h')
    dw = T.dot(v.T,g) / M
    dv = T.dot(g.T,h) / M
    da = T.mean(v, axis=0)
    db = T.mean(g, axis=0)
    dc = T.mean(h, axis=0)

    newgrads = lincg.linear_cg(
            lambda xw, xv, xa, xb, xc: natural.compute_Lx(vv,gg,hh,xw,xv,xa,xb,xc),
            [dw, dv, da, db, dc],
            rtol=1e-5,
            maxit = 30,
            damp = 0.,
            floatX = floatX,
            profile=0)

    f = theano.function([], newgrads)
    [new_dw, new_dv, new_da, new_db, new_dc] = f()
    numpy.testing.assert_almost_equal(Linv_x_w, new_dw, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_v, new_dv, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_a, new_da, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_b, new_db, decimal=1)
    numpy.testing.assert_almost_equal(Linv_x_c, new_dc, decimal=1)
