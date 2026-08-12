from open_r1.utils import import_utils


def test_transformers5_missing_package_tuple_is_false(monkeypatch):
    monkeypatch.setattr(import_utils, "_is_package_available", lambda _name: (False, None))
    assert import_utils._package_available("missing") is False


def test_legacy_boolean_package_probe_is_preserved(monkeypatch):
    monkeypatch.setattr(import_utils, "_is_package_available", lambda _name: True)
    assert import_utils._package_available("present") is True
