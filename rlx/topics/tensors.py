import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from IPython.display import display, Math, Latex


def show_latex(s):
    """
       example usage:

       > a,b=sy.symbols("a b")
       > expr = a+b
       > show_latex([["x =", expr], ["y =", expr**2]])

       displays in latex:

            x = a+b
            y = (a+b)^2

    """

    l = "\\begin{align}\n"
    for i in s:
        l+=i[0]+" \,&\, "+sy.latex(i[1])+"\\\\ \n"
    l += "\\end{align}"
    return Latex(l)


def build_metric_tensors(basis, only_covariant=False):
    r = sy.Matrix([[0]*len(basis)]*len(basis))
    for i, b1 in enumerate(basis):
        for j, b2 in enumerate(basis):
            r[i, j] = b1.dot(b2).simplify()
    if only_covariant:
        return r
    else:
        return r, r.inv()


def build_gradient(F, Z, Z_i):
    """
    F: the scalar field
    Z: the coordinate system symbols
    Z_i: the basis (should be the contravariant basis for the correct gradient)
    """
    k = sy.Matrix(Z_i).T * sy.Matrix([F.diff(i) for i in Z])
    k.simplify()
    return k


def build_contravariant_basis(Z_i, ZEij):
    """
    Z_i: the covariant basis
    ZEij: the contravariant metric tensor
    """
    r = ZEij*sy.Matrix(Z_i)
    return [r[i, :] for i in range(r.shape[0])]


def show_gradient(Zu, Zp, RZu, RZp):

    latex =([["\\textbf{position vectors}",""], ["\mathbf{R}(Z) =", RZu], ["\mathbf{R}(Z') =", RZp], ["", ""]])
    # the transformation equations
    fZu = RZp

    latex += [["\\textbf{transformation equations:}", ""]]
    latex += [["Z^i(Z'):", ""]]+[["&"+sy.latex(a(sy.Matrix(Zp).T))+"=", b] for a, b in zip(Zu, fZu)]

    Zu_i = build_covariant_basis(RZu, Zu)
    Zp_i = build_covariant_basis(RZp, Zp)
    Zu_ij, ZuEij = build_metric_tensors(Zu_i)
    Zp_ij, ZpEij = build_metric_tensors(Zp_i)
    ZpEi = build_contravariant_basis(Zp_i, ZpEij)
    ZuEi = build_contravariant_basis(Zu_i, ZuEij)



    latex += [["\\textbf{covariant basis:}", ""]] + \
             [["\mathbf{Z_%d} = \mathbf{Z_%s} = " % (c, Zu[c]), i] for c, i in enumerate(Zu_i)] + \
             [["", ""]] +\
             [["\mathbf{Z_{%d'}} = \mathbf{Z_{%s'}} = " % (c, Zp[c]), i] for c, i in enumerate(Zp_i)]

    latex += [["\\textbf{contravariant basis:}", ""]] + \
             [["\mathbf{Z^%d} = \mathbf{Z^%s} = "%(c,Zu[c]), i] for c, i in enumerate(ZuEi)] + \
             [["", ""]] +\
             [["\mathbf{Z^{%d'}} = \mathbf{Z^{%s'}} = "%(c,Zp[c]), i] for c, i in enumerate(ZpEi)]

    latex += [["", ""], ["\\textbf{metric tensors}", "(covariant\;contravariant\;primed\;unprimed)"]]
    latex += [["Z_{ij} = ", Zu_ij], ["Z^{ij} = ", ZuEij]]
    latex += [["Z'_{ij} = ", Zp_ij], ["Z'^{ij} = ", ZpEij]]


    Fu  = sum([i**2 for i in Zu])
    Fp  = Fu.subs({i: j for i, j in zip(Zu, fZu)})
    igu = build_gradient(Fu, Zu, ZuEi)

    latex += [["", ""], ["\\textbf{scalar function}", "in\;unprimed\;system"]]
    latex += [["F"+sy.latex(sy.Matrix(Zu).T)+"=", Fu]]
    latex += [["partial\;derivatives", sy.Matrix([Fu.diff(i) for i in Zu])]]
    latex += [["gradient", igu]]

    igp = build_gradient(Fp, Zp, ZuEi)
    latex += [["", ""], ["\\textbf{scalar function}", "in\;primed\;system"]]
    latex += [["F"+sy.latex(sy.Matrix(Zp).T)+"=", Fp]]
    latex += [["partial\; derivatives", sy.Matrix([Fp.diff(i) for i in Zp])]]
    latex += [["gradient", igp]]

    return show_latex(latex)


def plot_2D_transformation_grid(Zp, RZp, axis_equal=False):
    assert len(Zp) == 2
    _f = sy.lambdify(Zp, RZp, "numpy")

    z0_range = [0, 6]
    z1_range = [0, 6]

    z0_lines = range(int(z0_range[0]), int(z0_range[1])+1)
    z1_lines = range(int(z1_range[0]), int(z1_range[1])+1)

    for x in z0_lines:
        y = np.linspace(z0_range[0], z0_range[1], 100)
        eucl_coords = np.r_[[_f(i, j) for i, j in zip([x]*len(y), y)]][:, 0, :]
        plt.plot(eucl_coords[:, 0], eucl_coords[:, 1], color="red", alpha=.7)
    for y in z1_lines:
        x = np.linspace(z1_range[0], z1_range[1], 100)
        eucl_coords = np.r_[[_f(i, j) for i, j in zip(x, [y]*len(x))]][:, 0, :]
        plt.plot(eucl_coords[:, 0], eucl_coords[:, 1], color="red", alpha=.7)

    if axis_equal:
        plt.axis("equal")
    plt.grid()


def build_covariant_basis(RZ, Z):
    return [RZ.diff(i) for i in Z]


def plot_scalar_field(F, basis_vars=sy.symbols("x y"),
                      range0=(-1, 1), range1=(-1, 1), **kwargs):
    """
    assumes F is a sympy expression depending on basis_bars
    range0, range1: ranges for plotting each basis var
    """

    import itertools

    x = np.linspace(range0[0], range0[1], 20)
    y = np.linspace(range1[0], range1[1], 20)

    f_F = sy.lambdify(basis_vars, F, "numpy")

    Z = np.zeros((len(x), len(y)))
    for ix, iy in itertools.product(range(len(x)), range(len(y))):
        Z[ix, iy] = f_F(x[ix], y[iy])
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    plt.imshow(Z.T, origin="bottom", interpolation="bilinear", cmap=plt.cm.hot,
               extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
               **kwargs)
    plt.axis("equal")


def plot_vector_field(dF, basis_vars=sy.symbols("x y"), arrow_scale=.2,
                      range0=(-1, 1), range1=(-1, 1), **kwargs):
    """
    assumes dF is a matrix of two sympy expressions depending both on basis_bars
    range0, range1: ranges for plotting each basis var
    """

    import itertools

    x = np.linspace(range0[0], range0[1], 20)
    y = np.linspace(range1[0], range1[1], 20)

    f_dF = sy.lambdify(basis_vars, dF, "numpy")

    Z = np.zeros((2, len(x), len(y)))
    for ix, iy in itertools.product(range(len(x)), range(len(y))):
        Z[:, ix, iy] = f_dF(x[ix], y[iy])[:, 0]

    Z = Z/np.linalg.norm(Z, axis=0)/10  # shorten vectors for cosmetics
    for ix, iy in itertools.product(range(len(x)), range(len(y))):
        plt.arrow(x[ix], y[iy], Z[0, ix, iy], Z[1, ix, iy], **kwargs)

    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
