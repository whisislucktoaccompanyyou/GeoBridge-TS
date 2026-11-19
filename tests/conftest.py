from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from configuration import load_config
from training.smoke import smoke_config, make_mock_data


@pytest.fixture(scope='session', autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def base():
    config_dir = Path(__file__).resolve().parents[1]/'config'
    main = config_dir/'geobridge_ts.toml'
    return load_config(main if main.is_file() else config_dir/'geobridge_ts.example.toml')


@pytest.fixture
def mock_data(base, tmp_path):
    c = smoke_config(base, tmp_path)
    c['gateway']['request_url'] = 'http://127.0.0.1:9999/responses'
    rows, lookup = make_mock_data(c, tmp_path/'mock_inputs')
    return c, rows, lookup
