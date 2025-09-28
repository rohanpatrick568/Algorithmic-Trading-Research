from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    # Keep within Yahoo 1m API limit (<= 8 days per request) and ensure end < 'now' in exchange tz
    # Lumibot rejects a backtesting_end in the future; we clip to the previous completed minute.
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        ny_tz = ZoneInfo("America/New_York")
        now_ny = datetime.now(tz=ny_tz)
    except Exception:
        ny_tz = timezone.utc
        now_ny = datetime.now(tz=ny_tz)

    # Subtract a small safety buffer (2 minutes) and truncate to minute
    safe_end = now_ny - timedelta(minutes=2)
    safe_end = safe_end.replace(second=0, microsecond=0)
    end = safe_end
    start = end - timedelta(days=5)
    # Pick latest checkpoint if available (portable repo-relative path)
    import os, glob, pathlib, json
    default_dir = pathlib.Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir = os.environ.get("DS_CKPT_DIR", str(default_dir))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "bdq_*.pt")))
    # Prefer last.pt if exists for stable pointer
    last_pt = os.path.join(ckpt_dir, "last.pt")
    if os.path.exists(last_pt):
        model_path = last_pt
    else:
        model_path = ckpts[-1] if ckpts else os.path.join(ckpt_dir, "ckpt_test.pth")

    # Try to load manifest for inference params (fallback to defaults if absent)
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    price_bins = 21
    qty_bins = 11
    lookback = 120
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                mani = json.load(f)
            price_bins = int(mani.get("price_bins", price_bins))
            qty_bins = int(mani.get("qty_bins", qty_bins))
            lookback = int(mani.get("lookback", lookback))
        except Exception:
            pass

    strategy_cls = DeepScalperLumibotStrategy

    parameters = {
        "inference_cfg": InferenceConfig(
            model_path=model_path,
            price_bins=price_bins,
            qty_bins=qty_bins,
            lookback=lookback,
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
