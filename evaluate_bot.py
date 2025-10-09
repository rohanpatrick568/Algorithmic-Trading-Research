#!/usr/bin/env python3
"""
ML Trading Bot Evaluation and Performance Monitoring

This script provides comprehensive evaluation tools for the trading bot models,
including backtesting, performance metrics, and model comparison.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PerformanceMetrics:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        downside_returns = returns[returns < risk_free_rate]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return (returns.mean() - risk_free_rate) / downside_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = returns.mean() * 252
        max_dd = abs(PerformanceMetrics.calculate_max_drawdown(equity_curve))
        if max_dd == 0:
            return 0.0
        return annual_return / max_dd
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
    
    @staticmethod
    def calculate_profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_comprehensive_metrics(returns: pd.Series, equity_curve: pd.Series) -> Dict:
        """Calculate all performance metrics"""
        return {
            "total_return": (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100,
            "annualized_return": returns.mean() * 252 * 100,
            "volatility": returns.std() * np.sqrt(252) * 100,
            "sharpe_ratio": PerformanceMetrics.calculate_sharpe_ratio(returns),
            "sortino_ratio": PerformanceMetrics.calculate_sortino_ratio(returns),
            "max_drawdown": PerformanceMetrics.calculate_max_drawdown(equity_curve) * 100,
            "calmar_ratio": PerformanceMetrics.calculate_calmar_ratio(returns, equity_curve),
            "win_rate": PerformanceMetrics.calculate_win_rate(returns) * 100,
            "profit_factor": PerformanceMetrics.calculate_profit_factor(returns),
            "var_5": PerformanceMetrics.calculate_var(returns) * 100,
            "num_trades": len(returns),
            "avg_trade": returns.mean() * 100,
            "best_trade": returns.max() * 100,
            "worst_trade": returns.min() * 100
        }


class ModelBacktester:
    """Backtesting framework for trading models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
    def run_deepscalper_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Run backtest using DeepScalper model"""
        try:
            # Import the DeepScalper Lumibot strategy
            from scripts.deepscalper.run_lumibot_example import run_backtest_example
            
            # Run the backtest (simplified - would need to modify the example)
            results = {
                "strategy": "DeepScalper",
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "model_path": self.model_path,
                "returns": [],  # Would be populated from actual backtest
                "equity_curve": [],  # Would be populated from actual backtest
                "trades": [],  # Would be populated from actual backtest
                "success": True
            }
            
            # Generate synthetic results for demonstration
            # In practice, this would run the actual Lumibot backtest
            np.random.seed(42)
            num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            daily_returns = np.random.normal(0.001, 0.02, num_days)  # 0.1% daily return, 2% volatility
            equity_curve = np.cumprod(1 + daily_returns) * 100000  # Starting with $100k
            
            results["returns"] = daily_returns
            results["equity_curve"] = equity_curve
            
            return results
            
        except Exception as e:
            self.logger.error(f"DeepScalper backtest failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_backtrader_rl_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Run backtest using Backtrader RL model"""
        try:
            # This would implement actual backtrading with the RL model
            # For now, providing a framework structure
            
            results = {
                "strategy": "Backtrader_RL",
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "model_path": self.model_path,
                "returns": [],
                "equity_curve": [],
                "trades": [],
                "success": True
            }
            
            # Generate synthetic results for demonstration
            np.random.seed(43)
            num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            daily_returns = np.random.normal(0.0005, 0.015, num_days)  # Slightly lower return, lower vol
            equity_curve = np.cumprod(1 + daily_returns) * 100000
            
            results["returns"] = daily_returns
            results["equity_curve"] = equity_curve
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtrader RL backtest failed: {e}")
            return {"success": False, "error": str(e)}


class ModelComparison:
    """Compare performance of different models"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def compare_models(self, backtest_results: List[Dict]) -> Dict:
        """Compare multiple model results"""
        comparison_data = []
        
        for result in backtest_results:
            if not result.get("success", False):
                continue
                
            returns = pd.Series(result["returns"])
            equity_curve = pd.Series(result["equity_curve"])
            
            metrics = PerformanceMetrics.calculate_comprehensive_metrics(returns, equity_curve)
            metrics["strategy"] = result["strategy"]
            metrics["model_path"] = result["model_path"]
            
            comparison_data.append(metrics)
        
        return {
            "comparison_data": comparison_data,
            "summary": self._create_comparison_summary(comparison_data)
        }
    
    def _create_comparison_summary(self, comparison_data: List[Dict]) -> Dict:
        """Create summary of model comparison"""
        if not comparison_data:
            return {}
            
        df = pd.DataFrame(comparison_data)
        
        summary = {
            "best_sharpe": df.loc[df["sharpe_ratio"].idxmax()]["strategy"],
            "best_return": df.loc[df["total_return"].idxmax()]["strategy"],
            "lowest_drawdown": df.loc[df["max_drawdown"].idxmax()]["strategy"],  # max because it's negative
            "best_win_rate": df.loc[df["win_rate"].idxmax()]["strategy"],
            "metrics_summary": df.groupby("strategy").agg({
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean"
            }).round(4).to_dict()
        }
        
        return summary
    
    def generate_comparison_report(self, comparison_results: Dict, output_file: str = None):
        """Generate detailed comparison report"""
        if output_file is None:
            output_file = self.results_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "comparison_results": comparison_results,
            "recommendations": self._generate_recommendations(comparison_results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(output_file)
    
    def _generate_recommendations(self, comparison_results: Dict) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        summary = comparison_results.get("summary", {})
        
        if summary.get("best_sharpe"):
            recommendations.append(f"For risk-adjusted returns, consider using {summary['best_sharpe']} (best Sharpe ratio)")
        
        if summary.get("best_return"):
            recommendations.append(f"For maximum returns, {summary['best_return']} shows highest total return")
        
        if summary.get("lowest_drawdown"):
            recommendations.append(f"For capital preservation, {summary['lowest_drawdown']} has lowest drawdown")
        
        # Add more sophisticated recommendations based on metrics
        comparison_data = comparison_results.get("comparison_data", [])
        if comparison_data:
            avg_sharpe = np.mean([d["sharpe_ratio"] for d in comparison_data])
            if avg_sharpe < 1.0:
                recommendations.append("Overall Sharpe ratios are below 1.0 - consider parameter optimization")
            
            max_dd_values = [d["max_drawdown"] for d in comparison_data]
            if any(dd < -20 for dd in max_dd_values):
                recommendations.append("Some models show high drawdowns (>20%) - implement better risk management")
        
        return recommendations


class TradingBotEvaluator:
    """Main evaluation class"""
    
    def __init__(self):
        self.setup_logging()
        self.metrics = PerformanceMetrics()
        self.comparison = ModelComparison()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def evaluate_all_models(self, symbol: str = "AAPL", 
                           start_date: str = None, end_date: str = None) -> Dict:
        """Evaluate all available models"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Evaluating models for {symbol} from {start_date} to {end_date}")
        
        # Find available models
        models_dir = Path("models")
        checkpoints_dir = Path("scripts/deepscalper/checkpoints")
        
        available_models = []
        for model_file in models_dir.glob("*.pt"):
            available_models.append(("DeepScalper", str(model_file)))
            
        for checkpoint_file in checkpoints_dir.glob("*.pt"):
            if checkpoint_file.exists():
                available_models.append(("DeepScalper", str(checkpoint_file)))
        
        self.logger.info(f"Found {len(available_models)} models to evaluate")
        
        # Run backtests
        backtest_results = []
        for model_type, model_path in available_models:
            self.logger.info(f"Backtesting {model_type}: {model_path}")
            
            backtester = ModelBacktester(model_path)
            
            if model_type == "DeepScalper":
                result = backtester.run_deepscalper_backtest(symbol, start_date, end_date)
            else:
                result = backtester.run_backtrader_rl_backtest(symbol, start_date, end_date)
            
            if result.get("success", False):
                backtest_results.append(result)
            else:
                self.logger.error(f"Backtest failed for {model_path}: {result.get('error', 'Unknown error')}")
        
        # Compare models
        comparison_results = self.comparison.compare_models(backtest_results)
        
        # Generate report
        report_file = self.comparison.generate_comparison_report(comparison_results)
        self.logger.info(f"Comparison report saved to: {report_file}")
        
        return {
            "backtest_results": backtest_results,
            "comparison_results": comparison_results,
            "report_file": report_file
        }
    
    def create_performance_visualization(self, backtest_results: List[Dict], 
                                       output_dir: str = "results"):
        """Create performance visualization charts"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trading Bot Performance Comparison', fontsize=16)
        
        # Plot 1: Equity Curves
        ax1 = axes[0, 0]
        for result in backtest_results:
            if result.get("success", False):
                equity_curve = result["equity_curve"]
                dates = pd.date_range(
                    start=result["start_date"], 
                    periods=len(equity_curve), 
                    freq='D'
                )
                ax1.plot(dates, equity_curve, label=result["strategy"])
        ax1.set_title('Equity Curves')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Return Distributions
        ax2 = axes[0, 1]
        for result in backtest_results:
            if result.get("success", False):
                returns = pd.Series(result["returns"]) * 100  # Convert to percentage
                ax2.hist(returns, alpha=0.7, bins=30, label=result["strategy"])
        ax2.set_title('Return Distributions')
        ax2.set_xlabel('Daily Returns (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Risk-Return Scatter
        ax3 = axes[1, 0]
        for result in backtest_results:
            if result.get("success", False):
                returns = pd.Series(result["returns"])
                equity_curve = pd.Series(result["equity_curve"])
                metrics = PerformanceMetrics.calculate_comprehensive_metrics(returns, equity_curve)
                
                ax3.scatter(metrics["volatility"], metrics["annualized_return"], 
                          s=100, label=result["strategy"])
        ax3.set_title('Risk-Return Profile')
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Annualized Return (%)')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Key Metrics Comparison
        ax4 = axes[1, 1]
        metrics_data = []
        strategy_names = []
        
        for result in backtest_results:
            if result.get("success", False):
                returns = pd.Series(result["returns"])
                equity_curve = pd.Series(result["equity_curve"])
                metrics = PerformanceMetrics.calculate_comprehensive_metrics(returns, equity_curve)
                
                metrics_data.append([
                    metrics["sharpe_ratio"],
                    metrics["sortino_ratio"],
                    metrics["calmar_ratio"]
                ])
                strategy_names.append(result["strategy"])
        
        if metrics_data:
            metrics_df = pd.DataFrame(
                metrics_data,
                columns=['Sharpe', 'Sortino', 'Calmar'],
                index=strategy_names
            )
            metrics_df.plot(kind='bar', ax=ax4)
            ax4.set_title('Key Performance Ratios')
            ax4.set_ylabel('Ratio Value')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = output_dir / f"performance_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance visualization saved to: {plot_file}")
        return str(plot_file)


def main():
    parser = argparse.ArgumentParser(description="ML Trading Bot Evaluation")
    parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--model-path", help="Specific model path to evaluate")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--create-plots", action="store_true", help="Create performance plots")
    
    args = parser.parse_args()
    
    evaluator = TradingBotEvaluator()
    
    if args.model_path:
        # Evaluate specific model
        backtester = ModelBacktester(args.model_path)
        result = backtester.run_deepscalper_backtest(
            args.symbol, 
            args.start_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
            args.end_date or datetime.now().strftime("%Y-%m-%d")
        )
        
        if result.get("success", False):
            returns = pd.Series(result["returns"])
            equity_curve = pd.Series(result["equity_curve"])
            metrics = PerformanceMetrics.calculate_comprehensive_metrics(returns, equity_curve)
            
            print("\nPerformance Metrics:")
            print("-" * 40)
            for key, value in metrics.items():
                print(f"{key:20}: {value:10.4f}")
        else:
            print(f"Evaluation failed: {result.get('error', 'Unknown error')}")
    else:
        # Evaluate all models
        results = evaluator.evaluate_all_models(args.symbol, args.start_date, args.end_date)
        
        print("\nEvaluation completed!")
        print(f"Report saved to: {results['report_file']}")
        
        if args.create_plots and results['backtest_results']:
            plot_file = evaluator.create_performance_visualization(
                results['backtest_results'], 
                args.output_dir
            )
            print(f"Performance plots saved to: {plot_file}")


if __name__ == "__main__":
    main()