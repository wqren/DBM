import numpy
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

from DBM import natural
from DBM import minres
from DBM import lincg

rng = numpy.random.RandomState(92832)
(M,N0,N1,N2) = (1000,11,12,13)
v = rng.rand(M,N0).astype('float32')
g = rng.rand(M,N1).astype('float32')
h = rng.rand(M,N2).astype('float32')
xw_mat = (numpy.dot(v.T, g) / M).astype('float32')
xv_mat = (numpy.dot(g.T, h) / M).astype('float32')
xw = xw_mat.flatten()
xv = xv_mat.flatten()
x = numpy.hstack((xw.flatten(), xv.flatten()))

vg = (v[:,:,None] * g[:,None,:]).reshape(M,-1)
gh = (g[:,:,None] * h[:,None,:]).reshape(M,-1)
Lww = numpy.dot(vg.T, vg)
Lwv = numpy.dot(vg.T, gh)
Lvw = numpy.dot(gh.T, vg)
Lvv = numpy.dot(gh.T, gh)
Lw = numpy.dot(v.T, g).flatten()
Lv = numpy.dot(g.T, h).flatten()

L1 = numpy.vstack((
        numpy.hstack((Lww, Lwv)),
        numpy.hstack((Lvw, Lvv)))) / M
L1x = numpy.dot(L1, x)
L1x_w = L1x[:N0*N1].reshape(N0,N1)
L1x_v = L1x[N0*N1:].reshape(N1,N2)

L2 = numpy.outer(
            numpy.hstack((Lw, Lv)),
            numpy.hstack((Lw, Lv))) / M**2
L2x = numpy.dot(L2, x)
L2x_w = L2x[:N0*N1].reshape(N0,N1)
L2x_v = L2x[N0*N1:].reshape(N1,N2)

L = L1 - L2
Lx = numpy.dot(L, x)
Lx_w = Lx[:N0*N1].reshape(N0,N1)
Lx_v = Lx[N0*N1:].reshape(N1,N2)
## now compute L^-1 x
Linv_x = linalg.cho_solve(linalg.cho_factor(L), x)
Linv_xw = Linv_x[:N0*N1].reshape(N0,N1)
Linv_xv = Linv_x[N0*N1:].reshape(N1,N2)

def test_compute_Lx():

    ## now compare against theano version 
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    xxw_mat = T.matrix()
    xxv_mat = T.matrix()
    xxw = T.vector()
    xxv = T.vector()

    # just test that basic things run for now (shapes mostly)
    Lww_xw = natural.compute_Lww_xw(vv, gg, hh, xxw)
    f = theano.function([vv, gg, hh, xxw], Lww_xw, on_unused_input='ignore')
    rval1 = f(v, g, h, xw)

    Lwv_xv = natural.compute_Lwv_xv(vv, gg, hh, xxv)
    f = theano.function([vv, gg, hh, xxv], Lwv_xv, on_unused_input='ignore')
    rval2 = f(v, g, h, xv)

    Lvw_xw = natural.compute_Lvw_xw(vv, gg, hh, xxw)
    f = theano.function([vv, gg, hh, xxw], Lvw_xw, on_unused_input='ignore')
    rval3 = f(v, g, h, xw)

    Lvv_xv = natural.compute_Lvv_xv(vv, gg, hh, xxv)
    f = theano.function([vv, gg, hh, xxv], Lvv_xv, on_unused_input='ignore')
    rval4 = f(v, g, h, xv)

    # test compute_Lx_term1
    [Lx_term1_1, Lx_term1_2] = natural.compute_Lx_term1(vv, gg, hh, xxw, xxv)
    f = theano.function([vv, gg, hh, xxw, xxv], [Lx_term1_1, Lx_term1_2])
    [rval1, rval2] = f(v, g, h, xw, xv)
    numpy.testing.assert_almost_equal(L1x_w, rval1, decimal=3)
    numpy.testing.assert_almost_equal(L1x_v, rval2, decimal=3)

    # test compute_Lx_term2
    [Lx_term2_1, Lx_term2_2] = natural.compute_Lx_term2(vv, gg, hh, xxw, xxv)
    f = theano.function([vv, gg, hh, xxw, xxv], [Lx_term2_1, Lx_term2_2])
    [rval1, rval2] = f(v, g, h, xw, xv)
    numpy.testing.assert_almost_equal(L2x_w, rval1, decimal=3)

    # just test that basic things run for now (shapes mostly)
    Lww_xw = natural.compute_Lww_xw(vv, gg, hh, xxw)
    f = theano.function([vv, gg, hh, xxw], Lww_xw, on_unused_input='ignore')
    rval1 = f(v, g, h, xw)

    Lwv_xv = natural.compute_Lwv_xv(vv, gg, hh, xxv)
    f = theano.function([vv, gg, hh, xxv], Lwv_xv, on_unused_input='ignore')
    rval2 = f(v, g, h, xv)

    Lvw_xw = natural.compute_Lvw_xw(vv, gg, hh, xxw)
    f = theano.function([vv, gg, hh, xxw], Lvw_xw, on_unused_input='ignore')
    rval3 = f(v, g, h, xw)

    Lvv_xv = natural.compute_Lvv_xv(vv, gg, hh, xxv)
    f = theano.function([vv, gg, hh, xxv], Lvv_xv, on_unused_input='ignore')
    rval4 = f(v, g, h, xv)

    # test compute_Lx_term1
    [Lx_term1_1, Lx_term1_2] = natural.compute_Lx_term1(vv, gg, hh, xxw, xxv)
    f = theano.function([vv, gg, hh, xxw, xxv], [Lx_term1_1, Lx_term1_2])
    [rval1, rval2] = f(v, g, h, xw, xv)
    numpy.testing.assert_almost_equal(L1x_w, rval1, decimal=3)
    numpy.testing.assert_almost_equal(L1x_v, rval2, decimal=3)

    # test compute_Lx_term2
    [Lx_term2_1, Lx_term2_2] = natural.compute_Lx_term2(vv, gg, hh, xxw, xxv)
    f = theano.function([vv, gg, hh, xxw, xxv], [Lx_term2_1, Lx_term2_2])
    [rval1, rval2] = f(v, g, h, xw, xv)
    numpy.testing.assert_almost_equal(L2x_w, rval1, decimal=3)
    numpy.testing.assert_almost_equal(L2x_v, rval2, decimal=3)

    # test LLx
    LLx = natural.compute_Lx(vv, gg, hh, xxw_mat, xxv_mat)
    f = theano.function([vv, gg, hh, xxw_mat, xxv_mat], LLx)
    [rval1, rval2] = f(v, g, h, xw_mat, xv_mat)
    numpy.testing.assert_almost_equal(Lx_w, rval1, decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rval2, decimal=3)

 
def test_minres():
    (M,N0,N1,N2) = (10,11,12,13)
    
    vv = theano.shared(v, name='v')
    gg = theano.shared(g, name='g')
    hh = theano.shared(h, name='h')
    dw = T.dot(v.T,g) / M
    dv = T.dot(g.T,h) / M
    dw.name = 'dw'
    dv.name = 'dv'

    newgrads = minres.minres(
            lambda xw, xv: natural.compute_Lx(vv,gg,hh,xw,xv),
            [dw, dv],
            rtol=1e-5,
            damp = 0.,
            maxit = 100,
            profile=0)[0]
    f = theano.function([], newgrads)
    [new_dw, new_dv] = f()
    import pdb; pdb.set_trace()
    numpy.testing.assert_almost_equal(Linv_xw, new_dw, decimal=3)
    numpy.testing.assert_almost_equal(Linv_xv, new_dv, decimal=3)


def test_linearcg():
    (M,N0,N1,N2) = (10,11,12,13)
    
    vv = theano.shared(v, name='v')
    gg = theano.shared(g, name='g')
    hh = theano.shared(h, name='h')
    dw = T.dot(v.T,g) / M
    dv = T.dot(g.T,h) / M
    dw.name = 'dw'
    dv.name = 'dv'

    newgrads = lincg.linear_cg(
            lambda xw, xv: natural.compute_Lx(vv,gg,hh,xw,xv),
            [dw, dv],
            rtol=1e-4,
            maxit = 100,
            damp = 0.,
            floatX = floatX,
            profile=0)
    f = theano.function([], newgrads)
    [new_dw, new_dv] = f()
    import pdb; pdb.set_trace()
    numpy.testing.assert_almost_equal(Linv_xw, new_dw, decimal=3)
    numpy.testing.assert_almost_equal(Linv_xv, new_dv, decimal=3)
