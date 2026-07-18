import pytest
import argparse
from src.utils import parse_args


def test_parse_args(monkeypatch):
    monkeypatch.setenv('ENVIRONMENT', 'development')
    monkeypatch.setenv('JOB_TYPE', 'full_train')
    monkeypatch.setenv('DB_URI', 'https://www.google.com/')

    params = parse_args([])
    assert params.env == 'development'
    assert params.job_type == 'full_train'
    assert params.db_uri == 'https://www.google.com/'

def test_parse_args_type_exception(monkeypatch):
    monkeypatch.setenv('ENVIRONMENT', 'development')
    monkeypatch.setenv('JOB_TYPE', 'random_stuff')
    monkeypatch.setenv('DB_URI', 'https://www.google.com/')

    #ValueError, argparse.ArgumentError, SystemExit
    with pytest.raises(SystemExit) as excinfo:
        _ = parse_args([])

    assert isinstance(excinfo.value, SystemExit)
    arg_exception = excinfo.value.__context__
    assert isinstance(arg_exception, argparse.ArgumentError)
    value_exception = arg_exception.__context__
    assert isinstance(value_exception, ValueError)

def test_parse_args_missing_exception(monkeypatch):
    with pytest.raises(SystemExit) as excinfo:
        _ = parse_args([])

    assert isinstance(excinfo.value, SystemExit)
    arg_exception = excinfo.value.__context__
    assert isinstance(arg_exception, argparse.ArgumentError)
