"""Floating-point search for promising separable trial functions.

Discovery produces numerical candidates and diagnostics.  It must remain
independent of :mod:`maynard_tools.certification`: a successful numerical run
is useful evidence, but is not itself a rigorous certificate.
"""

__all__ = ["ratio"]
