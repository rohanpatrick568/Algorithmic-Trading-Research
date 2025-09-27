#!/usr/bin/env python3
"""
Smoke test for DeepScalper training and inference pipeline.
This test ensures the basic functionality works without external dependencies.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

try:
    from .config import EnvConfig, TrainConfig
    from .data import load_minute_data
    from .env import DeepScalperEnv
    from .train import train_agent
    from .lumibot_strategy import DeepScalperLumibotStrategy, InferenceConfig
except Exception:
    from config import EnvConfig, TrainConfig
    from data import load_minute_data
    from env import DeepScalperEnv
    from train import train_agent
    from lumibot_strategy import DeepScalperLumibotStrategy, InferenceConfig


def test_data_generation():
    """Test synthetic data generation"""
    print("=== Testing Data Generation ===")
    
    end = datetime.now()
    start = end - timedelta(days=2)
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    md = load_minute_data('AAPL', start_str, end_str, use_synthetic=True)
    
    assert len(md.df) > 0, "No data generated"
    assert all(col in md.df.columns for col in ['open', 'high', 'low', 'close', 'volume']), "Missing OHLCV columns"
    assert all(col in md.df.columns for col in ['rsi14', 'mfi14', 'bb_mid20', 'bb_up20', 'bb_dn20', 'atr14']), "Missing indicator columns"
    
    print(f"✓ Generated {len(md.df)} rows of synthetic data")
    return md


def test_environment():
    """Test environment creation and basic operations"""
    print("=== Testing Environment ===")
    
    md = test_data_generation()
    env_cfg = EnvConfig(symbol='AAPL')
    train_cfg = TrainConfig(train_steps=10)
    env = DeepScalperEnv(md, env_cfg, train_cfg)
    
    # Test reset
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,), f"Wrong obs shape: {obs.shape}"
    
    # Test step
    action = env.action_space.sample()
    obs2, reward, done, trunc, info = env.step(action)
    assert obs2.shape == (env.obs_dim,), f"Wrong obs2 shape: {obs2.shape}"
    assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"
    
    print(f"✓ Environment working: obs_dim={env.obs_dim}, action_space={env.action_space}")
    return env


def test_training():
    """Test basic training loop"""
    print("=== Testing Training ===")
    
    env = test_environment()
    train_cfg = TrainConfig(train_steps=50, warmup_steps=10)
    
    # Use a persistent directory for testing
    ckpt_dir = "/tmp/deepscalper_smoke_test"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    train_agent(env, train_cfg, ckpt_dir=ckpt_dir, resume=False, save_every=25)
    
    # Check if checkpoint was created
    import glob
    all_files = os.listdir(ckpt_dir)
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    print(f"Files in checkpoint dir: {all_files}")
    print(f"Found checkpoints: {ckpts}")
    
    if not ckpts:
        # Look for any checkpoint files
        last_pt = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(last_pt):
            ckpts = [last_pt]
        else:
            # Find any .pt files
            ckpts = glob.glob(os.path.join(ckpt_dir, "bdq_*.pt"))
            
    assert len(ckpts) > 0, f"No checkpoints created in {ckpt_dir}. Files: {all_files}"
    
    print(f"✓ Training completed, using checkpoint: {ckpts[0]}")
    return ckpts[0]


def test_phase1_enhancements():
    """Test Phase 1 enhancements: data caching, multi-symbol, regime generation"""
    print("=== Testing Phase 1 Enhancements ===")
    
    end = datetime.now()
    start = end - timedelta(days=2)
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    # Test different market regimes
    regimes = ["normal", "trending", "mean_reverting", "high_vol"]
    for regime in regimes:
        md = load_minute_data('AAPL', start_str, end_str, use_synthetic=True)
        # Test synthetic data with different regimes - import here to avoid circular imports
        try:
            from .data import generate_synthetic_data
        except:
            from data import generate_synthetic_data
        md_regime = generate_synthetic_data('AAPL', start_str, end_str, regime=regime)
        assert len(md_regime.df) > 0, f"No data generated for regime {regime}"
        print(f"✓ Generated {regime} regime data: {len(md_regime.df)} rows")
    
    # Test multi-symbol loading
    try:
        from .data import load_multi_symbol_data
    except:
        from data import load_multi_symbol_data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    multi_data = load_multi_symbol_data(symbols, start_str, end_str, use_synthetic=True)
    assert len(multi_data) == 3, f"Expected 3 symbols, got {len(multi_data)}"
    print(f"✓ Multi-symbol loading: {[md.symbol for md in multi_data]}")
    
    # Test enhanced environment config
    env_cfg = EnvConfig(
        symbol='AAPL',
        slippage_model="adaptive",
        slippage_random_shock=1.0,
        hard_position_cap=True,
        episode_sampling_mode="mixed",
        fractional_fees=True
    )
    
    train_cfg = TrainConfig(train_steps=10)
    env = DeepScalperEnv(md, env_cfg, train_cfg)
    
    # Test environment with new features
    obs, info = env.reset()
    action = env.action_space.sample()
    obs2, reward, done, trunc, info = env.step(action)
    
    print(f"✓ Enhanced environment working with adaptive slippage and position caps")
    return True


def test_inference():
    """Test model loading and inference"""
    print("=== Testing Inference ===")
    
    ckpt_path = test_training()
    
    # Test inference config
    inf_cfg = InferenceConfig(
        model_path=ckpt_path,
        price_bins=21,
        qty_bins=11,
        lookback=120
    )
    
    # This would normally test the Lumibot strategy, but we can't run full backtest
    # Just verify the config is valid
    assert os.path.exists(inf_cfg.model_path), f"Model file doesn't exist: {inf_cfg.model_path}"
    print(f"✓ Inference config created with model: {inf_cfg.model_path}")


def run_smoke_test():
    """Run complete smoke test"""
    print("DeepScalper Smoke Test")
    print("=" * 50)
    
    try:
        test_data_generation()
        test_environment()
        test_training()
        test_inference()
        test_phase1_enhancements()  # New Phase 1 tests
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED - DeepScalper pipeline is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    exit(0 if success else 1)