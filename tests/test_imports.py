"""Package-level imports must not activate optional certification backends."""


def test_top_level_packages_import() -> None:
    import maynard_tools
    import maynard_tools.certification
    import maynard_tools.discovery
    import maynard_tools.verifier

    import neuracert

    assert maynard_tools.__version__ == "0.9.0"
    assert neuracert.__version__ == "0.9.0"
