import pytest
import argparse
from src.utils import parse_args


def test_parse_args(monkeypatch):
    monkeypatch.setenv('ENVIRONMENT', 'Development')
    monkeypatch.setenv('JOB_TYPE', 'full_train')
    monkeypatch.setenv('DB_URI', 'https://www.google.com/')

    params = parse_args([])
    assert params.env == 'Development'
    assert params.job_type == 'full_train'
    assert params.db_uri == 'https://www.google.com/'

def test_parse_args_type_exception(monkeypatch):
    # Job Type missmatch
    monkeypatch.setenv('ENVIRONMENT', 'Development')
    monkeypatch.setenv('JOB_TYPE', 'random_job')
    monkeypatch.setenv('DB_URI', 'https://www.google.com/')

    with pytest.raises(SystemExit) as excinfo:
        _ = parse_args([])

        assert isinstance(excinfo.value, SystemExit)
        arg_exception = excinfo.value.__context__
        assert isinstance(arg_exception, argparse.ArgumentError)
        value_exception = arg_exception.__context__
        assert isinstance(value_exception, ValueError)

    # Environment missmatch
    monkeypatch.setenv('JOB_TYPE', 'full_train')
    monkeypatch.setenv('ENVIRONMENT', 'random_env')

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
