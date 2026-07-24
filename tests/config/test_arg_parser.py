import pytest
import argparse
from src.config.arg_parser import parse_args


def test_parse_args(monkeypatch):
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    monkeypatch.setenv('ENVIRONMENT', 'Development')
    monkeypatch.setenv('JOB_TYPE', 'full_train')
    monkeypatch.setenv('DB_URI', 'cover_lancedb')
    monkeypatch.setenv('EMBED_LAMBDA_NAME', 'embed_lambda')
    monkeypatch.setenv('MODEL_ROOT_DIR', 'model_path/')
    monkeypatch.setenv('TOWER_DIM', '64')

    params = parse_args([])
    assert params.env == 'Development'
    assert params.job_type == 'full_train'
    assert params.db_uri == 'cover_lancedb'

def test_parse_args_type_exception(monkeypatch):
    # Job Type missmatch
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    monkeypatch.setenv('ENVIRONMENT', 'Development')
    monkeypatch.setenv('JOB_TYPE', 'random_job')
    monkeypatch.setenv('DB_URI', 'cover_lancedb')
    monkeypatch.setenv('EMBED_LAMBDA_NAME', 'embed_lambda')
    monkeypatch.setenv('MODEL_ROOT_DIR', 'model_path/')
    monkeypatch.setenv('TOWER_DIM', '64')

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

    # Float type missmatch
    monkeypatch.setenv('ENVIRONMENT', 'Development')
    monkeypatch.setenv('USER_LR', 'not_a_number')

    with pytest.raises(SystemExit) as excinfo:
        _ = parse_args([])

        assert isinstance(excinfo.value, SystemExit)
        arg_exception = excinfo.value.__context__
        assert isinstance(arg_exception, argparse.ArgumentError)
        value_exception = arg_exception.__context__
        assert isinstance(value_exception, ValueError)

    # Integer type missmatch
    monkeypatch.setenv('USER_LR', '0.001')
    monkeypatch.setenv('TOWER_DIM', '3.14')

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
