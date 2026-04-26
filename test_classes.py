import pytest
from ml_runner import MLExperiment, RegularisedExperiment

class TestMLExperiment:
    """Test for the MLExperiment class."""

    def test_creation_default_lr(self):
        """Experiment should use 0.01 as default learning rate."""
        exp = MLExperiment('baseline')
        assert exp.name == 'baseline'
        assert exp.learning_rate == 0.01
        assert exp.results == []

    def test_creation_custon_lr(self):
        """Experiment should store the learning rate we pass."""
        exp = MLExperiment('fast', learning_rate=0.1)
        assert exp.learning_rate == 0.1

    def test_run_returns_correct_length(self):
        """Results list should have one entry per epoch."""
        exp = MLExperiment('test')
        results = exp.run(epochs=5)
        assert len(results) == 5
        assert len(exp) == 5

    def test_run_loss_decreases(self):
        """Each epoch should have lower loss then the previous one."""
        exp = MLExperiment('test')
        exp.run(epochs=5)
        for i in range(len(exp.results) - 1):
            assert exp.results[i] > exp.results[i + 1]

    def test_best_result_after_run(self):
        """best_result() shoul return the minimun loss."""
        exp = MLExperiment('test')
        exp.run(epochs=5)
        assert exp.best_result() == min(exp.results)

    def test_result_before_run_raises(self):
        """best_result() should raise ValueError if not yet run."""
        exp = MLExperiment('test')
        with pytest.raises(ValueError, match="Run the experiment first"):
            exp.best_result()
    
    def test_repr_contains_name(self):
        """__repr__ should include the experiment name."""
        exp = MLExperiment('my-run')
        assert 'my-run' in repr(exp)

    def test_total_count_increments(self):
        """Class counter should track all experiments created"""
        before = MLExperiment.total_experiments
        MLExperiment('count-test-1')
        MLExperiment('count-test-2')
        assert MLExperiment.total_experiments == before + 2

class TestRegularisedExperiment:
    """Tests for the RegularisedExperiment subclass."""

    def test_is_subclass(self):
        """RegularisedExperiment must be an MLExperiment."""
        reg = RegularisedExperiment('reg')
        assert isinstance(reg, MLExperiment)

    def test_regularisation_adds_penalty(self):
        """Regularised results should always be higher than base."""
        base = MLExperiment('base', learning_rate=0.01)
        reg = RegularisedExperiment('reg', learning_rate=0.01,
                                    lambda_reg=0.01)
        base.run(4)
        reg.run(4)
        for b, r in zip(base.results, reg.results):
            assert r > b

