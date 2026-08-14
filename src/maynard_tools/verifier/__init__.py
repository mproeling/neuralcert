"""Independent verification tools.

Modules in this package intentionally reimplement every required operation and
must not import from :mod:`maynard_tools.discovery` or
:mod:`maynard_tools.certification`.  This makes the verifier an independent
check even though it ships in the same distribution for user convenience.
"""

__all__ = ["certificate", "direct_ij"]

