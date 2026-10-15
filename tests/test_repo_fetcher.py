import pytest

from codeatlas.services.repo_fetcher import (
    InvalidRepoUrl, parse_github_url, repo_key, validate_repo_id,
)


@pytest.mark.parametrize("url, expected", [
    ("https://github.com/pallets/flask", ("pallets", "flask")),
    ("https://github.com/pallets/flask/", ("pallets", "flask")),
    ("https://github.com/pallets/flask.git", ("pallets", "flask")),
    ("  https://github.com/psf/requests  ", ("psf", "requests")),
    ("https://github.com/Textualize/rich", ("Textualize", "rich")),
    ("https://github.com/a-b/some.repo_name-1", ("a-b", "some.repo_name-1")),
])
def test_valid_urls(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize("url", [
    "", "flask", "http://github.com/pallets/flask",
    "https://gitlab.com/pallets/flask", "https://github.com/pallets",
    "https://github.com/pallets/flask/tree/main",
    "git@github.com:pallets/flask.git", "file:///etc/passwd",
    "https://github.com/pallets/flask --upload-pack=evil",
    "https://github.com/--upload-pack=x/flask",
    "https://github.com/pallets/..", "https://github.com/pallets/.",
    "https://github.com/../flask", "https://github.com/pallets/flask?x=1",
    "https://evil.com/https://github.com/pallets/flask",
])
def test_invalid_urls_are_rejected(url):
    with pytest.raises(InvalidRepoUrl):
        parse_github_url(url)


def test_repo_id_validation_blocks_path_tricks():
    for owner, name in [("..", "x"), ("a", ".."), ("a/b", "c"), ("a", "b/c"), ("a", "...")]:
        with pytest.raises(InvalidRepoUrl):
            validate_repo_id(owner, name)


def test_repo_key_is_case_insensitive():
    assert repo_key("Pallets", "Flask") == repo_key("pallets", "flask")
