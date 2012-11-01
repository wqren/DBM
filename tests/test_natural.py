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
    import pdb; pdb.set_trace()
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
