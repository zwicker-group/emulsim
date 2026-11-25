"""Simple implementation of Bessel functions for numba compilation

These implementations have been derived using OpenAI GPT-5.6 Terra and are automatically
tested against the :mod:`scipy` implementations.

.. autosummary::
   :nosignatures:

   ~i0
   ~i1
   ~k0
   ~k1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import math

import numba as nb


@nb.njit(inline="always")
def _poly(x, coefficients):
    """Evaluate c[0] + c[1]*x + ... using Horner's scheme."""
    result = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        result = result * x + coefficients[i]
    return result


@nb.njit
def i0(x):
    """Modified Bessel function I_0(x)."""
    ax = abs(x)

    if ax < 3.75:
        y = (x / 3.75) ** 2
        return 1.0 + y * _poly(
            y,
            (
                3.5156229,
                3.0899424,
                1.2067492,
                0.2659732,
                0.0360768,
                0.0045813,
            ),
        )

    y = 3.75 / ax
    return (
        math.exp(ax)
        / math.sqrt(ax)
        * _poly(
            y,
            (
                0.39894228,
                0.01328592,
                0.00225319,
                -0.00157565,
                0.00916281,
                -0.02057706,
                0.02635537,
                -0.01647633,
                0.00392377,
            ),
        )
    )


@nb.njit
def i1(x):
    """Modified Bessel function I_1(x)."""
    ax = abs(x)

    if ax < 3.75:
        y = (x / 3.75) ** 2
        result = ax * _poly(
            y,
            (
                0.5,
                0.87890594,
                0.51498869,
                0.15084934,
                0.02658733,
                0.00301532,
                0.00032411,
            ),
        )
    else:
        y = 3.75 / ax
        result = (
            math.exp(ax)
            / math.sqrt(ax)
            * _poly(
                y,
                (
                    0.39894228,
                    -0.03988024,
                    -0.00362018,
                    0.00163801,
                    -0.01031555,
                    0.02282967,
                    -0.02895312,
                    0.01787654,
                    -0.00420059,
                ),
            )
        )

    return -result if x < 0.0 else result


@nb.njit
def k0(x):
    """Modified Bessel function K_0(x), defined for x > 0."""
    if x <= 0.0:
        return math.inf

    if x <= 2.0:
        y = 0.25 * x * x
        return -math.log(0.5 * x) * i0(x) + _poly(
            y,
            (
                -0.57721566,
                0.42278420,
                0.23069756,
                0.03488590,
                0.00262698,
                0.00010750,
                0.00000740,
            ),
        )

    y = 2.0 / x
    return (
        math.exp(-x)
        / math.sqrt(x)
        * _poly(
            y,
            (
                1.25331414,
                -0.07832358,
                0.02189568,
                -0.01062446,
                0.00587872,
                -0.00251540,
                0.00053208,
            ),
        )
    )


@nb.njit
def k1(x):
    """Modified Bessel function K_1(x), defined for x > 0."""
    if x <= 0.0:
        return math.inf

    if x <= 2.0:
        y = 0.25 * x * x
        return (
            math.log(0.5 * x) * i1(x)
            + _poly(
                y,
                (
                    1.0,
                    0.15443144,
                    -0.67278579,
                    -0.18156897,
                    -0.01919402,
                    -0.00110404,
                    -0.00004686,
                ),
            )
            / x
        )

    y = 2.0 / x
    return (
        math.exp(-x)
        / math.sqrt(x)
        * _poly(
            y,
            (
                1.25331414,
                0.23498619,
                -0.03655620,
                0.01504268,
                -0.00780353,
                0.00325614,
                -0.00068245,
            ),
        )
    )
