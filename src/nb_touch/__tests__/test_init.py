import nb_touch


def test_version_is_set():
    assert isinstance(nb_touch.__version__, str)
    assert nb_touch.__version__ != ""
