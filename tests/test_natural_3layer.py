import numpy
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

from DBM import natural
from DBM import minres
from DBM import lincg

rng = numpy.random.RandomState(92832)
(M,N0,N1,N2,N3) = (256,3,4,5,6)
v = rng.randint(low=0, high=2, size=(M,N0)).astype('float32')
g = rng.randint(low=0, high=2, size=(M,N1)).astype('float32')
h = rng.randint(low=0, high=2, size=(M,N2)).astype('float32')
q = rng.randint(low=0, high=2, size=(M,N3)).astype('float32')
xw_mat = (numpy.dot(v.T, g) / M).astype('float32')
xv_mat = (numpy.dot(g.T, h) / M).astype('float32')
xz_mat = (numpy.dot(h.T, q) / M).astype('float32')
xw = xw_mat.flatten()
xv = xv_mat.flatten()
xz = xz_mat.flatten()
xa = numpy.mean(v, axis=0)
xb = numpy.mean(g, axis=0)
xc = numpy.mean(h, axis=0)
xd = numpy.mean(q, axis=0)
x = numpy.hstack((xw, xv, xz, xa, xb, xc, xd))

vg = (v[:,:,None] * g[:,None,:]).reshape(M,-1)
gh = (g[:,:,None] * h[:,None,:]).reshape(M,-1)
hq = (h[:,:,None] * q[:,None,:]).reshape(M,-1)

# W-rows
Lww = numpy.dot(vg.T, vg)
Lwv = numpy.dot(vg.T, gh)
Lwz = numpy.dot(vg.T, hq)
Lwa = numpy.dot(vg.T, v)
Lwb = numpy.dot(vg.T, g)
Lwc = numpy.dot(vg.T, h)
Lwd = numpy.dot(vg.T, q)
# V-rows
Lvw = numpy.dot(gh.T, vg)
Lvv = numpy.dot(gh.T, gh)
Lvz = numpy.dot(gh.T, hq)
Lva = numpy.dot(gh.T, v)
Lvb = numpy.dot(gh.T, g)
Lvc = numpy.dot(gh.T, h)
Lvd = numpy.dot(gh.T, q)
# Z-rows
Lzw = numpy.dot(hq.T, vg)
Lzv = numpy.dot(hq.T, gh)
Lzz = numpy.dot(hq.T, hq)
Lza = numpy.dot(hq.T, v)
Lzb = numpy.dot(hq.T, g)
Lzc = numpy.dot(hq.T, h)
Lzd = numpy.dot(hq.T, q)
# a-rows
Law = numpy.dot(v.T, vg)
Lav = numpy.dot(v.T, gh)
Laz = numpy.dot(v.T, hq)
Laa = numpy.dot(v.T, v)
Lab = numpy.dot(v.T, g)
Lac = numpy.dot(v.T, h)
Lad = numpy.dot(v.T, q)
# b-rows
Lbw = numpy.dot(g.T, vg)
Lbv = numpy.dot(g.T, gh)
Lbz = numpy.dot(g.T, hq)
Lba = numpy.dot(g.T, v)
Lbb = numpy.dot(g.T, g)
Lbc = numpy.dot(g.T, h)
Lbd = numpy.dot(g.T, q)
# c-rows
Lcw = numpy.dot(h.T, vg)
Lcv = numpy.dot(h.T, gh)
Lcz = numpy.dot(h.T, hq)
Lca = numpy.dot(h.T, v)
Lcb = numpy.dot(h.T, g)
Lcc = numpy.dot(h.T, h)
Lcd = numpy.dot(h.T, q)
# d-rows
Ldw = numpy.dot(q.T, vg)
Ldv = numpy.dot(q.T, gh)
Ldz = numpy.dot(q.T, hq)
Lda = numpy.dot(q.T, v)
Ldb = numpy.dot(q.T, g)
Ldc = numpy.dot(q.T, h)
Ldd = numpy.dot(q.T, q)

Lw = numpy.dot(v.T, g).flatten()
Lv = numpy.dot(g.T, h).flatten()
Lz = numpy.dot(h.T, q).flatten()
# mean will be taken later
La = numpy.sum(v, axis=0)
Lb = numpy.sum(g, axis=0)
Lc = numpy.sum(h, axis=0)
Ld = numpy.sum(q, axis=0)

Minv = numpy.float32(1./M)
M2inv = numpy.float32(1./M**2)

##### 2-layer stuff #####
L1 = numpy.vstack((
        numpy.hstack((Lww, Lwv, Lwz, Lwa, Lwb, Lwc, Lwd)),
        numpy.hstack((Lvw, Lvv, Lvz, Lva, Lvb, Lvc, Lvd)),
        numpy.hstack((Lzw, Lzv, Lzz, Lza, Lzb, Lzc, Lzd)),
        numpy.hstack((Law, Lav, Laz, Laa, Lab, Lac, Lad)),
        numpy.hstack((Lbw, Lbv, Lbz, Lba, Lbb, Lbc, Lbd)),
        numpy.hstack((Lcw, Lcv, Lcz, Lca, Lcb, Lcc, Lcd)),
        numpy.hstack((Ldw, Ldv, Ldz, Lda, Ldb, Ldc, Ldd)),
        )) * Minv

L2 = numpy.outer(
            numpy.hstack((Lw, Lv, Lz, La, Lb, Lc, Ld)),
            numpy.hstack((Lw, Lv, Lz, La, Lb, Lc, Ld))) * M2inv

L = L1 - L2
Lx = numpy.dot(L, x)
Lx_w = Lx[:N0*N1].reshape(N0,N1)
Lx_v = Lx[N0*N1 : N0*N1 + N1*N2].reshape(N1,N2)
Lx_z = Lx[N0*N1 + N1*N2 : N0*N1 + N1*N2 + N2*N3].reshape(N2,N3)
Lx_a = Lx[-N3-N2-N1-N0:-N3-N2-N1]
Lx_b = Lx[-N3-N2-N1:-N3-N2]
Lx_c = Lx[-N3-N2:-N3]
Lx_d = Lx[-N3:]

## now compute L^-1 x
Linv_x = linalg.cho_solve(linalg.cho_factor(L), x)
Linv_x_w = Linv_x[:N0*N1].reshape(N0,N1)
Linv_x_v = Linv_x[N0*N1 : N0*N1 + N1*N2].reshape(N1,N2)
Linv_x_z = Linv_x[N0*N1 + N1*N2 : N0*N1 + N1*N2 + N2*N3].reshape(N2,N3)
Linv_x_a = Linv_x[-N3-N2-N1-N0:-N3-N2-N1]
Linv_x_b = Linv_x[-N3-N2-N1:-N3-N2]
Linv_x_c = Linv_x[-N3-N2:-N3]
Linv_x_d = Linv_x[-N3:]


def test_generic_compute_Lx_batches():

    ## now compare against theano version
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    qq = T.matrix()
    aa = T.vector()
    bb = T.vector()
    cc = T.vector()
    dd = T.vector()
    xxw_mat = T.matrix()
    xxv_mat = T.matrix()
    xxz_mat = T.matrix()
    xxw = T.vector()
    xxv = T.vector()
    xxa = T.vector()
    xxb = T.vector()
    xxc = T.vector()
    xxd = T.vector()

    # test compute_Lx
    LLx = natural.generic_compute_Lx_batches([vv, gg, hh, qq],
                                             [xxw_mat, xxv_mat, xxz_mat],
                                             [xxa, xxb, xxc, xxd],
                                             256, 64)
    f = theano.function([vv, gg, hh, qq, xxw_mat, xxv_mat, xxz_mat, xxa, xxb, xxc, xxd], LLx)
    rvals = f(v, g, h, q, xw_mat, xv_mat, xz_mat, xa, xb, xc, xd)
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_z, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[4], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[5], decimal=3)
    numpy.testing.assert_almost_equal(Lx_d, rvals[6], decimal=3)


def test_generic_compute_Lx():

    ## now compare against theano version
    vv = T.matrix()
    gg = T.matrix()
    hh = T.matrix()
    qq = T.matrix()
    aa = T.vector()
    bb = T.vector()
    cc = T.vector()
    dd = T.vector()
    xxw_mat = T.matrix()
    xxv_mat = T.matrix()
    xxz_mat = T.matrix()
    xxw = T.vector()
    xxv = T.vector()
    xxa = T.vector()
    xxb = T.vector()
    xxc = T.vector()
    xxd = T.vector()

    # test compute_Lx
    LLx = natural.generic_compute_Lx([vv, gg, hh, qq],
                                     [xxw_mat, xxv_mat, xxz_mat],
                                     [xxa, xxb, xxc, xxd])
    f = theano.function([vv, gg, hh, qq, xxw_mat, xxv_mat, xxz_mat, xxa, xxb, xxc, xxd], LLx)
    rvals = f(v, g, h, q, xw_mat, xv_mat, xz_mat, xa, xb, xc, xd)
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_z, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[4], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[5], decimal=3)
    numpy.testing.assert_almost_equal(Lx_d, rvals[6], decimal=3)
