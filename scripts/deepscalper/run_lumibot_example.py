from __future__ import annotations

from datetime import datetime, timedelta

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

# Support running as a module or as a script
try:
    from .lumibot_strategy import DeepScalperLumibotStrategy, InferenceConfig
except Exception:
    import importlib, pathlib, sys
    root = str(pathlib.Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.append(root)
    mod = importlib.import_module("scripts.deepscalper.lumibot_strategy")
    DeepScalperLumibotStrategy = getattr(mod, "DeepScalperLumibotStrategy")
    InferenceConfig = getattr(mod, "InferenceConfig")


def main():
    symbol = "AAPL"
    asset = Asset(symbol, asset_type="stock")
    # Keep within Yahoo 1m API limit (<= 8 days per request)
    end = datetime.now()
    start = end - timedelta(days=5)
    # Pick latest checkpoint if available
    import os, glob, pathlib
    # Use repo-relative checkpoints dir (created by training) instead of user-specific absolute path
    default_ckpt_dir = pathlib.Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir = os.environ.get("DS_CKPT_DIR", str(default_ckpt_dir))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "bdq_*.pt")))
    model_path = ckpts[-1] if ckpts else os.path.join(ckpt_dir, "ckpt_test.pth")

    strategy_cls = DeepScalperLumibotStrategy

    parameters = {
        "inference_cfg": InferenceConfig(
            model_path=model_path,
            price_bins=21,
            qty_bins=11,
            lookback=120,
        ),
        "symbols": [symbol],
        "backtest_window": (start, end),
    }
    # Use the Strategy.backtest helper to ensure correct date windowing with Yahoo 1m
    strategy_cls.backtest(
        YahooDataBacktesting,
        start,
        end,
        parameters=parameters,
        benchmark_asset=asset,
    )


if __name__ == "__main__":
    main()
