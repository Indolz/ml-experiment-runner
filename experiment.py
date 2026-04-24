import time
import functools

def timer(func):
    """Decorator: prints how long a function took to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        enlapsed = time.time() - start
        print(f" [{func.__name__}] took {enlapsed:.4f}s")
        return result
    return wrapper
    
class MLExperiment:
    """Tracks a single ML training run."""

    # Class variable - shared by ALL instances
    total_experiments = 0

    def __init__(self, name: str, learning_rate: float = 0.01):
        #Instance variables - unique to each object
        self.name = name
        self.learning_rate = learning_rate
        self.results = []
        MLExperiment.total_experiments += 1

    @timer
    def run(self, epochs: int) -> list:
        """Simulate a training run."""
        self.results = [round(1 / (i + 1) * self.learning_rate, 4) for i in range(epochs)]
        return self.results
    
    def best_result(self) -> float:
        """Return the lowest loss achieved.
        
        Raises:
            ValueError: if the experimente has not been run yet.
        """
        if not self.results:
            raise ValueError("Run the experiment first.")
        return min(self.results)
    
    def __repr__(self) -> str:
        return f"MLExperiment(name={self.name!r}, lr={self.learning_rate})"
    
    def __len__(self) -> int:
        return len(self.results)
    
class RegularisedExperiment(MLExperiment):
    """An experiment with L2 regularisation penalty applied."""
    def __init__(self, name: str, learning_rate: float = 0.01, lambda_reg: float = 0.001):
        super().__init__(name, learning_rate) # run parent's __init__
        self.lambda_reg = lambda_reg

    def run(self, epochs: int) -> list:
        """Override parent run - apply regularisation penalty."""
        base_results = super().run(epochs) # get parent's results
        self.results = [round(r + self.lambda_reg, 4) for r in base_results]
        return self.results
    
    def __repr__(self) -> str:
        return (f"RegularisedExperiment(name={self.name!r}, "
                f"lr={self.learning_rate}, λ={self.lambda_reg})")
    

