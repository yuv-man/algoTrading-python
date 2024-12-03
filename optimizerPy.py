from scipy.optimize import minimize

class Optimizer:
    def __init__(self, backtester):
        self.backtester = backtester

    def optimize(self, strategy, param_ranges, initial_guess):
        """
        Optimizes the parameters of a strategy using `scipy.optimize`.

        Parameters:
        strategy (function): The strategy function to optimize.
        param_ranges (dict): Parameter names and bounds as (min, max).
        initial_guess (list): Initial guesses for the parameters.

        Returns:
        dict: Optimal parameters and backtest performance.
        """
        def objective(params):
            param_dict = {name: params[i] for i, name in enumerate(param_ranges.keys())}
            result = self.backtester.backtest(strategy, **param_dict)
            return -result['Net Profit']  # Minimize negative profit (maximize profit)

        bounds = [param_ranges[name] for name in param_ranges.keys()]
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

        optimal_params = {name: result.x[i] for i, name in enumerate(param_ranges.keys())}
        optimal_performance = self.backtester.backtest(strategy, **optimal_params)

        return {
            "Optimal Parameters": optimal_params,
            "Optimal Performance": optimal_performance,
        }
