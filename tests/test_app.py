"""Route-level smoke tests for the Flask app."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as app_module  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app_module.app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///" + str(tmp_path / "t.db"),
    )
    with app_module.app.app_context():
        app_module.db.engine.dispose()
        app_module.db.create_all()
    with app_module.app.test_client() as c:
        yield c


@pytest.mark.parametrize("path", ["/", "/abstract", "/accuracy", "/login", "/register"])
def test_public_pages(client, path):
    assert client.get(path).status_code == 200


def test_upload_requires_login(client):
    r = client.get("/upload")
    assert r.status_code in (301, 302)
    assert "/login" in r.headers["Location"]


def test_register_and_login(client):
    r = client.post("/register", data={
        "username": "tester", "email": "t@example.com",
        "password": "Str0ngPass!", "confirm_password": "Str0ngPass!",
    }, follow_redirects=True)
    assert r.status_code == 200
