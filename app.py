"""
GP Crypto Trading Strategy - Dashboard
=======================================
Streamlit UI for genetic programming crypto trading strategy.

Run:  streamlit run app.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from gp_crypto_evolution import (
    PAIRS, PRIMARY_PAIR, ARG_NAMES,
    INITIAL_CASH, COMMISSION_PCT, NO_TRADE_BAND, TIMEFRAME,
    POP_SIZE, N_GEN, P_CX, P_MUT, MAX_DEPTH, MAX_LEN,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    MODELS_DIR, DATA_DIR,
    load_all_pairs, split_dataset, vectorized_backtest,
    run_evolution, select_best_on_validation, backtest_on_slice,
    evaluate_individual, toolbox, fetch_klines,
    get_feature_arrays,
)

st.set_page_config(
    page_title="GP Crypto Strategy",
    page_icon="🧬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "df_all" not in st.session_state:
    st.session_state.df_all = None
if "hof" not in st.session_state:
    st.session_state.hof = None
if "best_ind" not in st.session_state:
    st.session_state.best_ind = None
if "evolution_log" not in st.session_state:
    st.session_state.evolution_log = []


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Parameters
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🧬 GP Parameters")

preset = st.sidebar.selectbox(
    "Preset",
    ["Custom", "Quick Test", "Balanced (Default)", "Deep Search", "Paper-grade"],
    index=2,
)

preset_map = {
    "Quick Test":          {"pop": 500,  "gen": 5,  "depth": 6, "maxlen": 40},
    "Balanced (Default)":  {"pop": 2000, "gen": 15, "depth": 8, "maxlen": 60},
    "Deep Search":         {"pop": 5000, "gen": 25, "depth": 8, "maxlen": 80},
    "Paper-grade":         {"pop": 7500, "gen": 30, "depth": 8, "maxlen": 60},
}

if preset != "Custom":
    p = preset_map[preset]
    default_pop, default_gen = p["pop"], p["gen"]
    default_depth, default_maxlen = p["depth"], p["maxlen"]
else:
    default_pop, default_gen = POP_SIZE, N_GEN
    default_depth, default_maxlen = MAX_DEPTH, MAX_LEN

st.sidebar.subheader("Evolution")
pop_size = st.sidebar.slider("Population Size", 100, 10000, default_pop, 100)
n_gen = st.sidebar.slider("Generations", 1, 50, default_gen)
p_cx = st.sidebar.slider("Crossover Prob", 0.5, 1.0, P_CX, 0.05)
p_mut = st.sidebar.slider("Mutation Prob", 0.01, 0.50, P_MUT, 0.01)
max_depth = st.sidebar.slider("Max Tree Depth", 3, 12, default_depth)
max_len = st.sidebar.slider("Max Tree Nodes", 10, 120, default_maxlen, 5)

st.sidebar.subheader("Trading")
commission = st.sidebar.select_slider(
    "Commission (per side)",
    options=[0.0001, 0.0002, 0.0004, 0.0006, 0.001],
    value=COMMISSION_PCT,
    format_func=lambda x: f"{x*100:.2f}%",
)
dead_band = st.sidebar.slider("Dead Band (pp)", 1, 30, NO_TRADE_BAND)
initial_cash = st.sidebar.number_input(
    "Initial Cash ($)", 10000, 10_000_000, INITIAL_CASH, 10000,
)

st.sidebar.subheader("Data Periods")
primary_pair = st.sidebar.selectbox("Primary Pair", PAIRS, index=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading market data...")
def cached_load_data():
    return load_all_pairs()


def compute_equity_curve(individual, df_slice):
    """Compute equity curve for charting."""
    func = toolbox.compile(expr=individual)
    cols = get_feature_arrays(df_slice, PRIMARY_PAIR)
    desired_pcts = func(*cols)
    desired_pcts = np.where(np.isfinite(desired_pcts), desired_pcts, 0.0)
    desired_pcts = np.clip(desired_pcts, -100.0, 100.0)
    weights = desired_pcts / 100.0

    delta = np.abs(np.diff(weights, prepend=0.0))
    weights[delta < dead_band / 100.0] = np.nan
    weights = pd.Series(weights).ffill().fillna(0.0).values

    close = df_slice[f"{PRIMARY_PAIR}_close"].to_numpy(dtype="float64")
    price_ret = np.diff(close) / close[:-1]
    strat_ret = weights[:-1] * price_ret
    turnover = np.abs(np.diff(weights, prepend=0.0))
    costs = turnover[:-1] * commission * 2
    net_ret = strat_ret - costs

    equity = initial_cash * np.cumprod(1 + net_ret)
    buy_hold = initial_cash * (close[1:] / close[0])

    idx = df_slice.index[1:]
    return pd.DataFrame({
        "GP Strategy": equity,
        "Buy & Hold": buy_hold,
    }, index=idx), weights, desired_pcts


def display_metrics(result, label=""):
    """Display metrics in columns."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Return", f"{result['total_return']*100:+.2f}%")
    c2.metric("Sharpe", f"{result['sharpe']:.3f}")
    c3.metric("Max DD", f"{result['max_drawdown']*100:.2f}%")
    c4.metric("Trades", f"{result['n_trades']}")
    c5.metric("Final Equity", f"${result['final_equity']:,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧬 GP Crypto Trading Strategy")

tab_evolve, tab_backtest, tab_signal, tab_docs = st.tabs([
    "🔬 Evolution", "📊 Backtest", "📡 Live Signal", "📖 Guide",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Evolution
# ═══════════════════════════════════════════════════════════════════════════
with tab_evolve:
    st.header("GP Evolution")

    col_info, col_action = st.columns([2, 1])
    with col_info:
        est_time = pop_size * n_gen / 2000 * 15 / 15  # rough estimate in seconds
        st.info(
            f"**Settings:** pop={pop_size:,}, gen={n_gen}, "
            f"cx={p_cx}, mut={p_mut}, depth={max_depth}, len={max_len}\n\n"
            f"**Estimated time:** ~{max(est_time, 1):.0f} seconds"
        )

    with col_action:
        run_btn = st.button("🚀 Start Evolution", type="primary", use_container_width=True)

    if run_btn:
        # Load data
        df_all = cached_load_data()
        st.session_state.df_all = df_all
        train_df, val_df, test_df = split_dataset(df_all)

        st.write(f"**Data:** {len(df_all):,} bars | "
                 f"Train: {len(train_df):,} | Val: {len(val_df):,} | "
                 f"Test: {len(test_df):,}")

        # Override globals for this run
        import gp_crypto_evolution as gce
        gce.COMMISSION_PCT = commission
        gce.NO_TRADE_BAND = dead_band
        gce.INITIAL_CASH = initial_cash
        gce.MAX_DEPTH = max_depth
        gce.MAX_LEN = max_len
        gce.P_CX = p_cx
        gce.P_MUT = p_mut

        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()
        log_container = st.empty()

        # Run evolution with progress capture
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            hof = run_evolution(train_df, pop_size=pop_size, n_gen=n_gen)

        log_output = buf.getvalue()
        progress_bar.progress(80, text="Selecting best on validation...")

        with redirect_stdout(io.StringIO()) as buf2:
            best = select_best_on_validation(hof, val_df)

        progress_bar.progress(90, text="Testing on out-of-sample...")

        test_result = vectorized_backtest(
            test_df[f"{PRIMARY_PAIR}_close"].to_numpy(dtype="float64"),
            toolbox.compile(expr=best)(
                *get_feature_arrays(test_df, PRIMARY_PAIR)
            ),
            initial_cash=initial_cash,
            commission=commission,
            dead_band=dead_band,
        )

        progress_bar.progress(100, text="Complete!")

        st.session_state.hof = hof
        st.session_state.best_ind = best

        # Results
        st.subheader("Test Results (Out-of-Sample)")
        display_metrics(test_result)

        # Equity curve
        eq_df, weights, signals = compute_equity_curve(best, test_df)
        st.subheader("Equity Curve (Test Period)")
        st.line_chart(eq_df)

        # Signal distribution
        col_sig, col_wt = st.columns(2)
        with col_sig:
            st.subheader("Signal Distribution")
            sig_df = pd.DataFrame({"signal": signals})
            st.bar_chart(sig_df["signal"].value_counts(bins=50).sort_index())
        with col_wt:
            st.subheader("Weight Over Time")
            wt_df = pd.DataFrame({"weight": weights}, index=test_df.index)
            st.line_chart(wt_df)

        # GP tree expression
        st.subheader("GP Tree Expression")
        st.code(str(best), language="python")
        st.caption(f"Tree size: {len(best)} nodes | "
                   f"Fitness: {best.fitness.values[0]:.6f}")

        # Evolution log
        with st.expander("Evolution Log"):
            st.text(log_output)

        # Save model
        import dill
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "best_crypto_gp.dill"
        with open(model_path, "wb") as f:
            dill.dump(best, f)
        st.success(f"Model saved: {model_path}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Backtest
# ═══════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("Backtest Saved Model")

    model_files = list(MODELS_DIR.glob("*.dill")) if MODELS_DIR.exists() else []

    if not model_files:
        st.warning("No saved models found. Run Evolution first.")
    else:
        selected_model = st.selectbox(
            "Model File",
            model_files,
            format_func=lambda p: p.name,
        )

        col_start, col_end = st.columns(2)
        bt_start = col_start.text_input("Start Date", TEST_START)
        bt_end = col_end.text_input("End Date", TEST_END)

        if st.button("📊 Run Backtest", type="primary"):
            import dill

            with open(selected_model, "rb") as f:
                model = dill.load(f)

            st.write(f"**Model:** {selected_model.name} | "
                     f"Tree: {len(model)} nodes | "
                     f"Fitness: {model.fitness.values[0]:.6f}")
            st.code(str(model), language="python")

            df_all = cached_load_data()
            df_slice = df_all.loc[bt_start:bt_end]

            if df_slice.empty:
                st.error("No data for selected period.")
            else:
                st.write(f"**Period:** {bt_start} ~ {bt_end} ({len(df_slice):,} bars)")

                func = toolbox.compile(expr=model)
                cols = get_feature_arrays(df_slice, PRIMARY_PAIR)
                desired_pcts = func(*cols)
                close = df_slice[f"{PRIMARY_PAIR}_close"].to_numpy(dtype="float64")

                result = vectorized_backtest(
                    close, desired_pcts,
                    initial_cash=initial_cash,
                    commission=commission,
                    dead_band=dead_band,
                )

                display_metrics(result)

                eq_df, weights, signals = compute_equity_curve(model, df_slice)
                st.subheader("Equity Curve")
                st.line_chart(eq_df)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Weight Over Time")
                    st.line_chart(
                        pd.DataFrame({"weight": weights}, index=df_slice.index)
                    )
                with col_b:
                    st.subheader(f"{PRIMARY_PAIR} Price")
                    st.line_chart(
                        pd.DataFrame(
                            {"price": close}, index=df_slice.index
                        )
                    )

        # Walk-forward section
        st.divider()
        st.subheader("Walk-Forward Analysis")

        wf_window = st.slider("Window (months)", 1, 12, 3)
        wf_step = st.slider("Step (months)", 1, 6, 1)

        if st.button("📈 Run Walk-Forward"):
            import dill

            with open(selected_model, "rb") as f:
                model = dill.load(f)

            df_all = cached_load_data()
            results = []
            end = df_all.index[-1]
            current = df_all.index[0]

            progress = st.progress(0)
            total_windows = 0
            cur_check = current
            while cur_check + timedelta(days=wf_window * 30) <= end:
                total_windows += 1
                cur_check += timedelta(days=wf_step * 30)

            window_num = 0
            while current + timedelta(days=wf_window * 30) <= end:
                w_end = current + timedelta(days=wf_window * 30)
                window_df = df_all.loc[current:w_end]
                window_num += 1
                progress.progress(
                    window_num / max(total_windows, 1),
                    text=f"Window {window_num}/{total_windows}",
                )

                if len(window_df) >= 100:
                    func = toolbox.compile(expr=model)
                    cols = get_feature_arrays(window_df, PRIMARY_PAIR)
                    pcts = func(*cols)
                    close = window_df[
                        f"{PRIMARY_PAIR}_close"
                    ].to_numpy(dtype="float64")
                    r = vectorized_backtest(
                        close, pcts,
                        initial_cash=initial_cash,
                        commission=commission,
                        dead_band=dead_band,
                    )
                    results.append({
                        "Period": f"{current.strftime('%Y-%m-%d')} ~ "
                                  f"{w_end.strftime('%Y-%m-%d')}",
                        "Return (%)": round(r["total_return"] * 100, 2),
                        "Sharpe": round(r["sharpe"], 3),
                        "Max DD (%)": round(r["max_drawdown"] * 100, 2),
                        "Trades": r["n_trades"],
                    })

                current += timedelta(days=wf_step * 30)

            progress.progress(1.0, text="Complete!")

            if results:
                wf_df = pd.DataFrame(results)
                st.dataframe(wf_df, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                returns = [r["Return (%)"] for r in results]
                col1.metric("Avg Return", f"{np.mean(returns):+.2f}%")
                col2.metric(
                    "Win Rate",
                    f"{sum(1 for r in returns if r > 0)/len(returns)*100:.0f}%",
                )
                col3.metric("Worst DD", f"{min(r['Max DD (%)'] for r in results):.2f}%")

                st.bar_chart(
                    pd.DataFrame({"Return (%)": returns}),
                )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Live Signal
# ═══════════════════════════════════════════════════════════════════════════
with tab_signal:
    st.header("Live Trading Signal")

    model_files = list(MODELS_DIR.glob("*.dill")) if MODELS_DIR.exists() else []

    if not model_files:
        st.warning("No saved models found. Run Evolution first.")
    else:
        sig_model = st.selectbox(
            "Model", model_files,
            format_func=lambda p: p.name,
            key="sig_model",
        )

        if st.button("📡 Get Latest Signal", type="primary"):
            import dill

            with open(sig_model, "rb") as f:
                model = dill.load(f)

            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(hours=72)

            with st.spinner("Fetching latest market data..."):
                dfs = []
                for pair in PAIRS:
                    pair_df = fetch_klines(pair, TIMEFRAME, start_dt, end_dt)
                    pair_df.columns = [f"{pair}_{c}" for c in pair_df.columns]
                    dfs.append(pair_df)
                combined = pd.concat(dfs, axis=1).dropna()

            if combined.empty:
                st.error("No data available")
            else:
                func = toolbox.compile(expr=model)
                cols = get_feature_arrays(combined, PRIMARY_PAIR)
                signals = func(*cols)
                signals = np.where(np.isfinite(signals), signals, 0.0)
                signals = np.clip(signals, -100.0, 100.0)

                latest_signal = float(signals[-1])
                btc_price = float(
                    combined[f"{PRIMARY_PAIR}_close"].iloc[-1]
                )

                # Signal display
                if latest_signal > 10:
                    direction = "🟢 LONG"
                    color = "green"
                elif latest_signal < -10:
                    direction = "🔴 SHORT"
                    color = "red"
                else:
                    direction = "⚪ FLAT"
                    color = "gray"

                st.markdown(f"### {direction}")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("BTC Price", f"${btc_price:,.2f}")
                col2.metric("Signal", f"{latest_signal:+.1f}%")
                col3.metric("Strength", f"{abs(latest_signal):.0f}/100")
                col4.metric(
                    "Timestamp",
                    combined.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
                )

                # Signal history chart
                st.subheader("Signal History (Last 72h)")
                sig_hist = pd.DataFrame({
                    "Signal (%)": signals,
                    f"{PRIMARY_PAIR} Price": combined[
                        f"{PRIMARY_PAIR}_close"
                    ].values,
                }, index=combined.index)

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.line_chart(sig_hist["Signal (%)"])
                with col_chart2:
                    st.line_chart(sig_hist[f"{PRIMARY_PAIR} Price"])

                # Current market data table
                with st.expander("Current Market Data"):
                    latest_row = combined.iloc[-1]
                    market_data = {}
                    for pair in PAIRS:
                        market_data[pair] = {
                            "Open": f"${latest_row[f'{pair}_open']:,.2f}",
                            "High": f"${latest_row[f'{pair}_high']:,.2f}",
                            "Low": f"${latest_row[f'{pair}_low']:,.2f}",
                            "Close": f"${latest_row[f'{pair}_close']:,.2f}",
                        }
                    st.dataframe(pd.DataFrame(market_data).T)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Parameter Guide
# ═══════════════════════════════════════════════════════════════════════════
with tab_docs:
    guide_path = Path(__file__).parent / "docs" / "gp_parameters_guide.md"
    if guide_path.exists():
        st.markdown(guide_path.read_text())
    else:
        st.info("Parameter guide not found at docs/gp_parameters_guide.md")
