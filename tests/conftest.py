def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: loads the checkpoint and runs real sampling (minutes, needs a GPU)")
