import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def func(x, y):
    return (x + y) * np.exp(-5.0 * (x ** 2 + y ** 2))


y, x = np.mgrid[-1:1:15j, -1:1:15j]
fvals = func(x, y)
print(len(fvals[0]))

newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')

xnew = np.linspace(-1, 1, 100)
ynew = np.linspace(-1, 1, 100)
fnew = newfunc(xnew, ynew)

plt.subplot(121)
im1 = plt.imshow(fvals, extent=[-1, 1, -1, 1], cmap=plt.cm.hot, interpolation='nearest', origin='lower')
plt.colorbar(im1)

plt.subplot(122)
im2 = plt.imshow(fnew, extent=[-1, 1, -1, 1], cmap=plt.cm.hot, interpolation='nearest', origin='lower')
plt.colorbar(im2)

plt.show()
