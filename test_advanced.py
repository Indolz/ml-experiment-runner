import pytest
from unittest.mock import patch, MagicMock
from ml_runner import MLExperiment


# ── FIXTURES ──────────────────────────────────────────────────────
# A fixture is reusable setup code. Any test that lists it as a
# parameter gets it injected automatically by pytest.

@pytest.fixture
def fresh_experiment():
    """A baseline experiment, not yet run."""
    return MLExperiment("fixture-test", learning_rate=0.01)


@pytest.fixture
def completed_experiment():
    """An experiment that has already run 5 epochs."""
    exp = MLExperiment("completed", learning_rate=0.01)
    exp.run(epochs=5)
    return exp


# Tests that use fixtures — pytest sees the parameter name,
# finds the matching fixture, runs it, and passes the result in.

def test_fresh_has_no_results(fresh_experiment):
    assert fresh_experiment.results == []
    assert len(fresh_experiment) == 0


def test_completed_has_results(completed_experiment):
    assert len(completed_experiment) == 5
    assert completed_experiment.best_result() < 0.01


def test_best_result_is_minimum(completed_experiment):
    assert completed_experiment.best_result() == min(completed_experiment.results)


# ── PARAMETRIZE ───────────────────────────────────────────────────
# Run the same test with multiple different inputs.
# pytest generates one test case per tuple.

@pytest.mark.parametrize("learning_rate,epochs,expected_len", [
    (0.01, 3, 3),
    (0.1, 10, 10),
    (0.001, 1, 1),
])
def test_run_length_matches_epochs(learning_rate, epochs, expected_len):
    exp = MLExperiment("param-test", learning_rate=learning_rate)
    exp.run(epochs=epochs)
    assert len(exp) == expected_len


# ── MOCKING ───────────────────────────────────────────────────────
# Replace a real dependency with a fake one during the test.
# Use this when your code calls external things: APIs, files,
# databases, time — anything you can't control in a test.

def get_experiment_from_api(experiment_id: str) -> dict:
    """Simulates a function that fetches experiment data from an API."""
    import requests
    response = requests.get(f"https://api.example.com/experiments/{experiment_id}")
    return response.json()


def test_api_call_without_hitting_real_api():
    # Build a fake response object
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "exp-001",
        "name": "remote-experiment",
        "best_loss": 0.0042
    }

    # Replace requests.get with our fake — only inside this test
    with patch("requests.get", return_value=mock_response):
        result = get_experiment_from_api("exp-001")

    # Verify we got the fake data back
    assert result["name"] == "remote-experiment"
    assert result["best_loss"] == 0.0042
    # requests.get was never actually called — no network needed