
def test_placeholder():
    """
    Needed to make the test suite run and not throw
    an error about no tests being found when
    `make test_integrations` is used.
    
    The error would look like this:
        make: *** [Makefile:61: test_integrations] Error 5
    
    More information can be found here:
        https://github.com/pytest-dev/pytest/issues/2393
    """
    pass