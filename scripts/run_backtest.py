import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.engine import run_backtest

def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    res = run_backtest(cfg)
    # print quick summary
    ir_gross = res.pnl.mean() / res.pnl.std() * (12**0.5)
    ir_net   = res.pnl_net.mean() / res.pnl_net.std() * (12**0.5)
    print(f"IR gross: {ir_gross:.2f} | IR net: {ir_net:.2f}")
    print(f"Avg turnover: {res.turnover.mean():.2%} | Avg TE(ann): {res.te_ann.mean():.2%}")
    # save net active returns to output
    res.pnl_net.to_csv("output/active_net.csv")

if __name__ == "__main__":
    main()
