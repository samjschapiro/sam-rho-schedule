# Source: https://github.com/andim/projgrad/blob/master/projgrad/main.py
import cupy as np


def project_l2_ball(x, rho):
    """Project a vector onto an l2 norm ball of radius rho.

    Args:
        x (torch.Tensor): Input vector.
        rho (float): Radius of the l2 norm ball.

    Returns:
        torch.Tensor: Projected vector onto the l2 norm ball.
    """
    norm_x = np.linalg.norm(x, ord=2)  # Compute the l2 norm of the vector
    if norm_x <= rho:
        # If the norm of the vector is already within the radius, return the vector as it is
        return x
    else:
        # Otherwise, project the vector onto the l2 norm ball
        projected_x = x * (rho / norm_x)
        return projected_x

class OptimizeResult(dict):
    """ Represents the optimization result.

    Parameters
    ----------
    x : ndarray
        the solution of the optimization.
    success : bool
        whether or not the optimizer exited successfully.
    status : int
        termination status of the optimizer. its value depends on the
        underlying solver. refer to `message` for details.
    message : str
        description of the cause of the termination.
    fun, jac, hess, hess_inv : ndarray
        values of objective function, jacobian, hessian or its inverse (if
        available). the hessians may be approximations, see the documentation
        of the function in question.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

def minimize(fun, x0, args=(),
             project=project_l2_ball,
             nboundupdate=100,
             reltol=1e-4, abstol=0.0, maxiters=1e7,
             rho=0.05,
             algo='fast',
             disp=False,
             callback=None,
             mask=None):
    """
    minimize     fun(x)
    subject to constraints on x expressed through a projection

    parameters
    ----------
    fun : function returning cost, gradient
    x0 : starting guess
    args: Extra arguments passed to the objective function
    project : projection operator (default: projection onto probability simplex)
    nboundupdate : number of iteration between lower bound updates
    reltol, abstol, maxiters: numerical parameter
    algo: fast or normal algorithm
    disp: print status information during the run
    mask: Boolean array with directions along which not to optimize
    callback: callback function called as callback(f, p)

    output
    ------
    Optimal solution as `OptimizeResult` object. Parameters include 

    x : ndarray
        The solution of the optimization.
    fun : ndarray
        value of objective function

    """
    if mask is not None:
        mask = np.asarray(mask)

        def mfun(x, *args):
            f, grad = fun(x, *args)
            if grad is not None:
                grad[mask] = 0.0
            return f, grad
        mfun = mfun
        mproject = lambda p: project(p, mask)
    else:
        mfun = fun
        mproject = project
    # initialize p from function input
    p = mproject(np.asarray(x0), rho)
    # counting variable for number of iterations
    k = 0
    # lower bound for the cost function
    low = -np.inf

    # setup for accelerated algorithm
    if algo == 'fast':
        y = p
        f, grad = mfun(p, *args)
        # starting guess for gradient scaling parameter 1 / | nabla f |
        s = 1.0 / np.linalg.norm(grad)
        # refine by backtracking search
        n_iter_max = 50
        iter_num = 0
        while True:
            y_new = mproject(y - s * grad, rho)
            # abs important as values close to machine precision
            # might become negative in fft convolution screwing
            # up cost calculations
            f_new, grad_new = mfun(y_new, *args)
            if f_new < f + np.dot(y_new - y, grad.T) + \
                    0.5 * np.linalg.norm(y_new - y)**2 / s:
                break
            s *= 0.8
            iter_num += 1
            if iter_num == n_iter_max: break
        # reduce s by some factor as optimal s might become smaller during
        # the course of optimization
        s /= 3.0
    else:
        f, grad = mfun(p, *args)

    while k < maxiters:
        k += 1

        # update lower bound on cost function
        # initialize at beginning (k=1) and then every nboundupdateth iteration
        if (k % nboundupdate == 0) or (k == 1):
            if algo =='fast':
                f, grad = mfun(p, *args)
            if mask is not None:
                i = np.argmin(grad[~mask])
                low = max((low, f - np.sum(p * grad) + grad[~mask][i]))
            else:
                i = np.argmin(grad)
                low = max((low, f - np.sum(p * grad) + grad[i]))
            gap = f - low
            if callback:
                callback(f, p)
            if disp:
                print('%g: f %e, gap %e, relgap %e' % (k, f, gap, gap/low if low > 0 else np.inf))
            if ((low > 0 and gap/low < reltol) or gap < abstol):
                if disp:
                    print('stopping criterion reached')
                break

        if algo == 'fast':
            f, grad = mfun(y, *args)
            p, pold = mproject(y - s * grad, rho), p
            y = p + k/(k+3.0) * (p - pold)
        else:
            # see e.g section 4.2 in http://www.web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
            s = 1.0 / np.linalg.norm(grad)
            z = mproject(p - s * grad, rho)
            fnew, gradnew = mfun(z, *args)
            while fnew > f + np.dot(z - p, grad.T) + \
                    0.5 * np.linalg.norm(z - p)**2 / s:
                s *= 0.5
                z = mproject(p - s * grad, rho)
                fnew, gradnew = mfun(z, *args)
            p = z
            f, grad = fnew, gradnew
    if disp:
        print('cost %e, low %e, gap %e' % (f, low, gap))

    return OptimizeResult(x=p, fun=f, nit=k, success=True)
