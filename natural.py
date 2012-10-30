import numpy

import theano
import theano.tensor as T
floatX = theano.config.floatX

def compute_Lquad_x(s1, s2, s3, s4, x):
    """
    Computes one of the quadrants of the L matrix.
    """
    s1_star_s2 = (s1.dimshuffle(0,1,'x') * s2.dimshuffle(0,'x',1)).flatten(ndim=2)
    if s3==s1 and s4==s2:
        s3_star_s4 = s1_star_s2
    else:
        s3_star_s4 = (s3.dimshuffle(0,1,'x') * s4.dimshuffle(0,'x',1)).flatten(ndim=2)
    # s1_star_s2_x, vector of length M
    s3_star_s4_x = T.dot(s3_star_s4, x)
    # matrix of length N0*N1
    Lquad_x = T.dot(s1_star_s2.T, s3_star_s4_x)
    return Lquad_x

def compute_Lww_xw(v, g, h, x_w):
    return compute_Lquad_x(v, g, v, g, x_w)

def compute_Lwv_xv(v, g, h, x_v):
    return compute_Lquad_x(v, g, g, h, x_v)

def compute_Lvw_xw(v, g, h, x_w):
    return compute_Lquad_x(g, h, v, g, x_w)

def compute_Lvv_xv(v, g, h, x_v):
    return compute_Lquad_x(g, h, g, h, x_v)

def compute_Lx_term1(v, g, h, xw, xv):
    (N0, N1, N2) = (v.shape[1], g.shape[1], h.shape[1])
    M = T.cast(v.shape[0], floatX)
    Lx_layer0 = compute_Lww_xw(v, g, h, xw) + compute_Lwv_xv(v, g, h, xv)
    Lx_layer1 = compute_Lvw_xw(v, g, h, xw) + compute_Lvv_xv(v, g, h, xv)
    return [Lx_layer0.reshape((N0, N1)) / M,
            Lx_layer1.reshape((N1, N2)) / M]

def compute_Lw(v, g):
    return T.dot(v.T, g).flatten()

def compute_Lv(g, h):
    return T.dot(g.T, h).flatten()

def compute_Lx_term2(v, g, h, xw, xv):
    (N0, N1, N2) = (v.shape[1], g.shape[1], h.shape[1])
    M = T.cast(v.shape[0], floatX)
    Lw = compute_Lw(v,g).flatten()
    Lv = compute_Lw(g,h).flatten()
    Lw_xw_plus_Lx_xv = T.dot(Lw, xw) + T.dot(Lv, xv)
    Lx_layer0 = Lw * Lw_xw_plus_Lx_xv
    Lx_layer1 = Lv * Lw_xw_plus_Lx_xv
    return [Lx_layer0.reshape((N0, N1)) / M**2,
            Lx_layer1.reshape((N1, N2)) / M**2]

def compute_Lx(v, g, h, xw_mat, xv_mat):
    xw = xw_mat.flatten()
    xv = xv_mat.flatten()
    [Lx_term1_w, Lx_term1_v] = compute_Lx_term1(v, g, h, xw, xv)
    [Lx_term2_w, Lx_term2_v] = compute_Lx_term2(v, g, h,xw, xv)
    return [Lx_term1_w - Lx_term2_w,
            Lx_term1_v - Lx_term2_v]

