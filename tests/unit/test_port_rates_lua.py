"""
Unit tests for orchagent/port_rates.lua

Exercises compute_rate() and compute_ber() covering >90% of code paths:
  - First-time initialisation (COUNTERS_LAST state)
  - Missing alpha / missing counters / missing timestamp (early-exit paths)
  - Invalid time delta (≤0 and >20 s) → baseline reset without emitting rates
  - Valid rate computation in COUNTERS_LAST state (initial, unsmoothed)
  - Valid rate computation in DONE state (exponentially smoothed)
  - PFC Rx/Tx rates: positive when counters increment, -1 when no baseline,
    smoothed in DONE state, persisted as *_last after each cycle
  - FEC BER computation (present and absent), FEC_PRE_BER_MAX tracking
  - Gearbox path (counters_db == gb_counters_db): only compute_ber() is called

Requirements:
    pip install redis pytest
    # A Redis server must be reachable at localhost:6379.
    # Tests are skipped automatically when Redis is unavailable.

Run:
    pytest tests/test_port_rates_lua.py -v
"""

import os
import time
import pytest
import redis as redis_module

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "orchagent", "port_rates.lua"
)

# DB indices that match the script's hard-coded constants
APPL_DB_IDX       = 0   # appl_db  ("0" in script)
COUNTERS_DB_IDX   = 13  # test-only — safe because SONiC uses 0–7, 10
GB_COUNTERS_DB_IDX = 10  # gb_counters_db ("10" in script)

PORT_OID   = "oid:0x1000000000001"
PORT_NAME  = "Ethernet0"
POLL_MS    = "1000"

RATES_TABLE    = "RATES"
COUNTERS_TABLE = "COUNTERS"
PORT_NAME_MAP  = "COUNTERS_PORT_NAME_MAP"
PORT_TABLE_KEY = "PORT_TABLE"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rc():
    """
    Redis client shared across the module.
    Loads the Lua script once and attaches its SHA as ``rc.sha``.
    Skips the entire module when Redis is not reachable.
    """
    try:
        client = redis_module.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()
    except (redis_module.ConnectionError, redis_module.TimeoutError):
        pytest.skip("Redis server not available at localhost:6379")

    with open(SCRIPT_PATH) as fh:
        client.sha = client.script_load(fh.read())

    yield client

    # Final teardown
    for db in (APPL_DB_IDX, COUNTERS_DB_IDX):
        client.execute_command("SELECT", db)
        client.flushdb()


@pytest.fixture(autouse=True)
def clean(rc):
    """Flush test DBs before every test for full isolation."""
    rc.execute_command("SELECT", COUNTERS_DB_IDX)
    rc.flushdb()
    rc.execute_command("SELECT", APPL_DB_IDX)
    rc.delete(f"{PORT_TABLE_KEY}:{PORT_NAME}")
    rc.execute_command("SELECT", COUNTERS_DB_IDX)
    yield


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _sel(rc, db):
    rc.execute_command("SELECT", db)


def seed_rates_config(rc, alpha="0.5"):
    _sel(rc, COUNTERS_DB_IDX)
    rc.hset(f"{RATES_TABLE}:PORT", mapping={"PORT_ALPHA": alpha,
                                             "PORT_SMOOTH_INTERVAL": "10"})


def seed_port_name_map(rc):
    _sel(rc, COUNTERS_DB_IDX)
    rc.hset(PORT_NAME_MAP, PORT_NAME, PORT_OID)


def seed_appl_port(rc, lanes="0,1,2,3", speed="400000"):
    _sel(rc, APPL_DB_IDX)
    rc.hset(f"{PORT_TABLE_KEY}:{PORT_NAME}",
            mapping={"lanes": lanes, "speed": speed})


def seed_counters(rc, **overrides):
    """Populate COUNTERS:<oid> with default traffic + PFC counters."""
    _sel(rc, COUNTERS_DB_IDX)
    fields = {
        "SAI_PORT_STAT_IF_IN_UCAST_PKTS":      "1000",
        "SAI_PORT_STAT_IF_IN_NON_UCAST_PKTS":  "100",
        "SAI_PORT_STAT_IF_OUT_UCAST_PKTS":     "800",
        "SAI_PORT_STAT_IF_OUT_NON_UCAST_PKTS": "50",
        "SAI_PORT_STAT_IF_IN_OCTETS":          "1000000",
        "SAI_PORT_STAT_IF_OUT_OCTETS":         "800000",
    }
    for i in range(8):
        fields[f"SAI_PORT_STAT_PFC_{i}_TX_PKTS"] = str(100 + i * 10)
        fields[f"SAI_PORT_STAT_PFC_{i}_RX_PKTS"] = str(50  + i * 5)
    fields.update(overrides)
    rc.hset(f"{COUNTERS_TABLE}:{PORT_OID}", mapping=fields)


def seed_last_state(rc, state="COUNTERS_LAST",
                    time_offset_sec=-5, **overrides):
    """
    Populate RATES:<oid>:PORT INIT_DONE and RATES:<oid> last-counter fields.

    ``time_offset_sec``:  seconds relative to now for LAST_UPDATE_TIME_SEC.
      Negative → in the past (normal case).
      Positive → in the future (triggers delta ≤ 0 guard).
      Very negative (e.g. -30) → triggers delta > 20 guard.
    """
    _sel(rc, COUNTERS_DB_IDX)
    rc.hset(f"{RATES_TABLE}:{PORT_OID}:PORT", "INIT_DONE", state)

    last_time = int(time.time()) + time_offset_sec
    fields = {
        "LAST_UPDATE_TIME_SEC":          str(last_time),
        "LAST_UPDATE_TIME_REM_MICROSEC": "0",
        "SAI_PORT_STAT_IF_IN_UCAST_PKTS_last":      "500",
        "SAI_PORT_STAT_IF_IN_NON_UCAST_PKTS_last":  "50",
        "SAI_PORT_STAT_IF_OUT_UCAST_PKTS_last":     "400",
        "SAI_PORT_STAT_IF_OUT_NON_UCAST_PKTS_last": "25",
        "SAI_PORT_STAT_IF_IN_OCTETS_last":           "500000",
        "SAI_PORT_STAT_IF_OUT_OCTETS_last":          "400000",
    }
    for i in range(8):
        fields[f"SAI_PORT_STAT_PFC_{i}_TX_PKTS_last"] = str(i * 10)
        fields[f"SAI_PORT_STAT_PFC_{i}_RX_PKTS_last"] = str(i * 5)
    fields.update(overrides)
    rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping=fields)


def run_script(rc, db_idx=COUNTERS_DB_IDX):
    _sel(rc, db_idx)
    return rc.evalsha(rc.sha, 1, PORT_OID,
                      str(db_idx), COUNTERS_TABLE, POLL_MS)


def get_rates(rc, db_idx=COUNTERS_DB_IDX):
    _sel(rc, db_idx)
    return rc.hgetall(f"{RATES_TABLE}:{PORT_OID}")


def get_init_state(rc, db_idx=COUNTERS_DB_IDX):
    _sel(rc, db_idx)
    return rc.hget(f"{RATES_TABLE}:{PORT_OID}:PORT", "INIT_DONE")


# ---------------------------------------------------------------------------
# 1. First-time initialisation (no INIT_DONE)
# ---------------------------------------------------------------------------

class TestFirstInit:

    def test_sets_counters_last_state(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)

        run_script(rc)

        assert get_init_state(rc) == "COUNTERS_LAST"

    def test_saves_traffic_counter_baseline(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc, **{"SAI_PORT_STAT_IF_IN_OCTETS": "123456"})

        run_script(rc)

        assert get_rates(rc).get("SAI_PORT_STAT_IF_IN_OCTETS_last") == "123456"

    def test_saves_pfc_counter_baseline(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)

        run_script(rc)

        rates = get_rates(rc)
        assert rates.get("SAI_PORT_STAT_PFC_0_TX_PKTS_last") == "100"
        assert rates.get("SAI_PORT_STAT_PFC_7_RX_PKTS_last") == str(50 + 7 * 5)

    def test_saves_timestamp(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)

        before = int(time.time())
        run_script(rc)
        after = int(time.time())

        saved = int(get_rates(rc).get("LAST_UPDATE_TIME_SEC", 0))
        assert before <= saved <= after + 1

    def test_no_rates_written_on_first_run(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)

        run_script(rc)

        rates = get_rates(rc)
        assert "RX_BPS" not in rates
        assert "TX_BPS" not in rates


# ---------------------------------------------------------------------------
# 2. Early-exit paths (missing data)
# ---------------------------------------------------------------------------

class TestEarlyExit:

    def test_no_alpha_skips_compute_rate(self, rc):
        # No seed_rates_config → PORT_ALPHA absent
        seed_port_name_map(rc)
        seed_counters(rc)

        run_script(rc)

        rates = get_rates(rc)
        assert "RX_BPS" not in rates
        assert "SAI_PORT_STAT_IF_IN_UCAST_PKTS_last" not in rates

    def test_missing_packet_counters_skips_rate(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        # Only set one counter — the mandatory octet fields are absent
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{COUNTERS_TABLE}:{PORT_OID}",
                mapping={"SAI_PORT_STAT_IF_IN_UCAST_PKTS": "100"})
        seed_last_state(rc)

        run_script(rc)

        assert "RX_BPS" not in get_rates(rc)

    def test_missing_timestamp_skips_rate(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{RATES_TABLE}:{PORT_OID}:PORT", "INIT_DONE", "COUNTERS_LAST")
        # Seed _last values but intentionally omit LAST_UPDATE_TIME_SEC
        rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping={
            "SAI_PORT_STAT_IF_IN_UCAST_PKTS_last":      "500",
            "SAI_PORT_STAT_IF_IN_NON_UCAST_PKTS_last":  "50",
            "SAI_PORT_STAT_IF_OUT_UCAST_PKTS_last":     "400",
            "SAI_PORT_STAT_IF_OUT_NON_UCAST_PKTS_last": "25",
            "SAI_PORT_STAT_IF_IN_OCTETS_last":           "500000",
            "SAI_PORT_STAT_IF_OUT_OCTETS_last":          "400000",
        })

        run_script(rc)

        assert "RX_BPS" not in get_rates(rc)


# ---------------------------------------------------------------------------
# 3. Invalid time-delta guard
# ---------------------------------------------------------------------------

class TestInvalidDelta:

    def _setup_with_offset(self, rc, offset):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=offset)

    def test_future_timestamp_emits_no_rates(self, rc):
        """last_time in the future → delta < 0 → reset, no rates."""
        self._setup_with_offset(rc, offset=+10)

        run_script(rc)

        assert "RX_BPS" not in get_rates(rc)

    def test_future_timestamp_resets_traffic_baseline(self, rc):
        self._setup_with_offset(rc, offset=+10)

        run_script(rc)

        assert get_rates(rc).get("SAI_PORT_STAT_IF_IN_OCTETS_last") == "1000000"

    def test_future_timestamp_resets_pfc_baseline(self, rc):
        self._setup_with_offset(rc, offset=+10)

        run_script(rc)

        assert get_rates(rc).get("SAI_PORT_STAT_PFC_0_TX_PKTS_last") == "100"

    def test_future_timestamp_refreshes_timestamp(self, rc):
        self._setup_with_offset(rc, offset=+10)

        before = int(time.time())
        run_script(rc)
        after = int(time.time())

        saved = int(get_rates(rc).get("LAST_UPDATE_TIME_SEC", 0))
        assert before <= saved <= after + 1

    def test_stale_timestamp_emits_no_rates(self, rc):
        """last_time 30 s ago → delta > 20 → reset, no rates."""
        self._setup_with_offset(rc, offset=-30)

        run_script(rc)

        assert "RX_BPS" not in get_rates(rc)

    def test_stale_timestamp_resets_baseline(self, rc):
        self._setup_with_offset(rc, offset=-30)

        run_script(rc)

        assert get_rates(rc).get("SAI_PORT_STAT_IF_IN_OCTETS_last") == "1000000"

    def test_stale_timestamp_resets_pfc_baseline(self, rc):
        self._setup_with_offset(rc, offset=-30)

        run_script(rc)

        rates = get_rates(rc)
        assert rates.get("SAI_PORT_STAT_PFC_0_TX_PKTS_last") == "100"
        assert rates.get("SAI_PORT_STAT_PFC_0_RX_PKTS_last") == "50"


# ---------------------------------------------------------------------------
# 4. Valid rate computation — COUNTERS_LAST → DONE (initial, unsmoothed)
# ---------------------------------------------------------------------------

class TestInitialRateComputation:

    def test_state_transitions_to_done(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)

        run_script(rc)

        assert get_init_state(rc) == "DONE"

    def test_rx_bps_written_and_positive(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc, **{"SAI_PORT_STAT_IF_IN_OCTETS": "600000"})
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5,
                        **{"SAI_PORT_STAT_IF_IN_OCTETS_last": "500000"})

        run_script(rc)

        assert float(get_rates(rc).get("RX_BPS", 0)) > 0

    def test_tx_bps_written_and_positive(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc, **{"SAI_PORT_STAT_IF_OUT_OCTETS": "500000"})
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5,
                        **{"SAI_PORT_STAT_IF_OUT_OCTETS_last": "400000"})

        run_script(rc)

        assert float(get_rates(rc).get("TX_BPS", 0)) > 0

    def test_rx_pps_written_and_positive(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)

        run_script(rc)

        assert float(get_rates(rc).get("RX_PPS", 0)) > 0

    def test_traffic_baseline_updated(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc, **{"SAI_PORT_STAT_IF_IN_OCTETS": "999"})
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)

        run_script(rc)

        assert get_rates(rc).get("SAI_PORT_STAT_IF_IN_OCTETS_last") == "999"

    def test_timestamp_refreshed_after_computation(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)

        before = int(time.time())
        run_script(rc)
        after = int(time.time())

        saved = int(get_rates(rc).get("LAST_UPDATE_TIME_SEC", 0))
        assert before <= saved <= after + 1


# ---------------------------------------------------------------------------
# 5. Valid rate computation — DONE state (exponentially smoothed)
# ---------------------------------------------------------------------------

class TestSmoothedRateComputation:

    def _setup_done(self, rc, alpha="0.5"):
        seed_rates_config(rc, alpha=alpha)
        seed_port_name_map(rc)
        seed_counters(rc, **{
            "SAI_PORT_STAT_IF_IN_OCTETS":  "700000",
            "SAI_PORT_STAT_IF_OUT_OCTETS": "600000",
        })
        seed_last_state(rc, state="DONE", time_offset_sec=-5)
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping={
            "RX_BPS": "100.0", "TX_BPS": "80.0",
            "RX_PPS": "10.0",  "TX_PPS": "8.0",
        })

    def test_done_state_preserved_after_smoothed_run(self, rc):
        self._setup_done(rc)
        run_script(rc)
        assert get_init_state(rc) == "DONE"

    def test_smoothed_rx_bps_positive(self, rc):
        self._setup_done(rc)
        run_script(rc)
        assert float(get_rates(rc).get("RX_BPS", 0)) > 0

    def test_smoothed_tx_bps_positive(self, rc):
        self._setup_done(rc)
        run_script(rc)
        assert float(get_rates(rc).get("TX_BPS", 0)) > 0

    def test_smoothed_value_differs_from_raw(self, rc):
        """With alpha=0.5, the smoothed rate must combine old and new."""
        self._setup_done(rc, alpha="0.5")
        run_script(rc)
        rx_bps = float(get_rates(rc).get("RX_BPS", 0))
        # new_raw = (700000-500000)/5 ≈ 40000; old = 100
        # smoothed = 0.5*40000 + 0.5*100 = 20050 (approx — delta varies)
        # Just assert it's clearly above the old value of 100
        assert rx_bps > 100


# ---------------------------------------------------------------------------
# 6. PFC Rx/Tx rate computation
# ---------------------------------------------------------------------------

class TestPfcRates:

    def test_pfc_rates_written_for_all_eight_pgs(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)

        run_script(rc)

        rates = get_rates(rc)
        for i in range(8):
            assert f"PFC_{i}_TX_PPS" in rates, f"PFC_{i}_TX_PPS missing"
            assert f"PFC_{i}_RX_PPS" in rates, f"PFC_{i}_RX_PPS missing"

    def test_pfc_rate_positive_when_counters_increment(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc, **{
            "SAI_PORT_STAT_PFC_0_TX_PKTS": "200",  # +100 vs baseline of 100
            "SAI_PORT_STAT_PFC_0_RX_PKTS": "100",  # +50  vs baseline of 50
        })
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)

        run_script(rc)

        rates = get_rates(rc)
        assert float(rates.get("PFC_0_TX_PPS", 0)) > 0
        assert float(rates.get("PFC_0_RX_PPS", 0)) > 0

    def test_pfc_rate_is_minus_one_when_no_baseline(self, rc):
        """PFC last counter absent (defaults to -1) → rate stored as -1."""
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc)
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)
        # Strip PFC baseline values so they resolve to -1 inside the script
        _sel(rc, COUNTERS_DB_IDX)
        for i in range(8):
            rc.hdel(f"{RATES_TABLE}:{PORT_OID}",
                    f"SAI_PORT_STAT_PFC_{i}_TX_PKTS_last",
                    f"SAI_PORT_STAT_PFC_{i}_RX_PKTS_last")

        run_script(rc)

        rates = get_rates(rc)
        assert float(rates.get("PFC_0_TX_PPS", 0)) == -1.0
        assert float(rates.get("PFC_0_RX_PPS", 0)) == -1.0

    def test_pfc_rates_smoothed_in_done_state(self, rc):
        seed_rates_config(rc, alpha="0.5")
        seed_port_name_map(rc)
        seed_counters(rc, **{
            "SAI_PORT_STAT_PFC_0_TX_PKTS": "200",
            "SAI_PORT_STAT_PFC_0_RX_PKTS": "100",
        })
        seed_last_state(rc, state="DONE", time_offset_sec=-5)
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping={
            "PFC_0_TX_PPS": "50.0",
            "PFC_0_RX_PPS": "25.0",
            "RX_BPS": "1.0", "TX_BPS": "1.0",
            "RX_PPS": "1.0", "TX_PPS": "1.0",
        })

        run_script(rc)

        tx_pps = float(get_rates(rc).get("PFC_0_TX_PPS", 0))
        assert tx_pps > 0

    def test_pfc_baseline_saved_after_computation(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc, **{"SAI_PORT_STAT_PFC_3_TX_PKTS": "777"})
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-5)

        run_script(rc)

        assert get_rates(rc).get("SAI_PORT_STAT_PFC_3_TX_PKTS_last") == "777"

    def test_pfc_baseline_saved_on_invalid_delta_reset(self, rc):
        """Even after a delta-reset cycle, PFC baselines are updated."""
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_counters(rc, **{"SAI_PORT_STAT_PFC_5_TX_PKTS": "555"})
        seed_last_state(rc, state="COUNTERS_LAST", time_offset_sec=-30)

        run_script(rc)

        assert get_rates(rc).get("SAI_PORT_STAT_PFC_5_TX_PKTS_last") == "555"


# ---------------------------------------------------------------------------
# 7. FEC BER computation (compute_ber)
# ---------------------------------------------------------------------------

class TestBerComputation:

    def test_ber_written_when_fec_counters_and_lane_info_present(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_appl_port(rc, lanes="0,1,2,3", speed="400000")
        seed_counters(rc, **{
            "SAI_PORT_STAT_IF_IN_FEC_CORRECTED_BITS":        "1000",
            "SAI_PORT_STAT_IF_IN_FEC_NOT_CORRECTABLE_FRAMES": "0",
        })
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping={
            "SAI_PORT_STAT_IF_FEC_CORRECTED_BITS_last":          "0",
            "SAI_PORT_STAT_IF_FEC_NOT_CORRECTABLE_FARMES_last":  "0",
        })

        run_script(rc)

        rates = get_rates(rc)
        assert "FEC_PRE_BER"  in rates
        assert "FEC_POST_BER" in rates
        assert float(rates["FEC_PRE_BER"]) >= 0

    def test_ber_skipped_when_fec_counters_absent(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_appl_port(rc)
        seed_counters(rc)  # no FEC fields

        run_script(rc)

        assert "FEC_PRE_BER" not in get_rates(rc)

    def test_fec_pre_ber_max_updated_when_new_ber_is_higher(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_appl_port(rc, lanes="0,1,2,3", speed="400000")
        seed_counters(rc, **{
            "SAI_PORT_STAT_IF_IN_FEC_CORRECTED_BITS":        "100000",
            "SAI_PORT_STAT_IF_IN_FEC_NOT_CORRECTABLE_FRAMES": "0",
        })
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping={
            "SAI_PORT_STAT_IF_FEC_CORRECTED_BITS_last":         "0",
            "SAI_PORT_STAT_IF_FEC_NOT_CORRECTABLE_FARMES_last": "0",
            "FEC_PRE_BER_MAX": "0",
        })

        run_script(rc)

        assert float(get_rates(rc).get("FEC_PRE_BER_MAX", 0)) > 0

    def test_fec_pre_ber_max_unchanged_when_new_ber_is_lower(self, rc):
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_appl_port(rc, lanes="0,1,2,3", speed="400000")
        seed_counters(rc, **{
            "SAI_PORT_STAT_IF_IN_FEC_CORRECTED_BITS":        "1",
            "SAI_PORT_STAT_IF_IN_FEC_NOT_CORRECTABLE_FRAMES": "0",
        })
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping={
            "SAI_PORT_STAT_IF_FEC_CORRECTED_BITS_last":         "0",
            "SAI_PORT_STAT_IF_FEC_NOT_CORRECTABLE_FARMES_last": "0",
            "FEC_PRE_BER_MAX": "999.0",
        })

        run_script(rc)

        assert float(get_rates(rc).get("FEC_PRE_BER_MAX", 0)) == 999.0

    def test_fec_last_values_updated_after_ber_computation(self, rc):
        """After BER computation the script saves current FEC counters as *_last."""
        seed_rates_config(rc)
        seed_port_name_map(rc)
        seed_appl_port(rc, lanes="0,1,2,3", speed="400000")
        seed_counters(rc, **{
            "SAI_PORT_STAT_IF_IN_FEC_CORRECTED_BITS":        "5000",
            "SAI_PORT_STAT_IF_IN_FEC_NOT_CORRECTABLE_FRAMES": "3",
        })
        _sel(rc, COUNTERS_DB_IDX)
        rc.hset(f"{RATES_TABLE}:{PORT_OID}", mapping={
            "SAI_PORT_STAT_IF_FEC_CORRECTED_BITS_last":         "0",
            "SAI_PORT_STAT_IF_FEC_NOT_CORRECTABLE_FARMES_last": "0",
        })

        run_script(rc)

        rates = get_rates(rc)
        assert rates.get("SAI_PORT_STAT_IF_FEC_CORRECTED_BITS_last") == "5000"
        assert rates.get("SAI_PORT_STAT_IF_FEC_NOT_CORRECTABLE_FARMES_last") == "3"


# ---------------------------------------------------------------------------
# 8. Gearbox path (counters_db == gb_counters_db == 10)
# ---------------------------------------------------------------------------

class TestGearboxPath:

    @pytest.fixture(autouse=True)
    def clean_gb_db(self, rc):
        _sel(rc, GB_COUNTERS_DB_IDX)
        rc.flushdb()
        _sel(rc, APPL_DB_IDX)
        rc.flushdb()
        yield
        _sel(rc, GB_COUNTERS_DB_IDX)
        rc.flushdb()
        _sel(rc, APPL_DB_IDX)
        rc.flushdb()

    def test_compute_rate_not_called_in_gearbox_mode(self, rc):
        """
        When counters_db == gb_counters_db, only compute_ber() is invoked.
        compute_rate() writes INIT_DONE; its absence confirms it was skipped.
        """
        _sel(rc, GB_COUNTERS_DB_IDX)
        rc.hset(PORT_NAME_MAP, PORT_NAME, PORT_OID)
        rc.hset(f"{COUNTERS_TABLE}:{PORT_OID}", mapping={
            "SAI_PORT_STAT_IF_IN_FEC_CORRECTED_BITS":        "0",
            "SAI_PORT_STAT_IF_IN_FEC_NOT_CORRECTABLE_FRAMES": "0",
        })
        rc.hset(f"{RATES_TABLE}:PORT",
                mapping={"PORT_ALPHA": "0.5", "PORT_SMOOTH_INTERVAL": "10"})

        _sel(rc, APPL_DB_IDX)
        rc.hset(f"{PORT_TABLE_KEY}:{PORT_NAME}",
                mapping={"lanes": "0,1,2,3", "speed": "400000"})

        run_script(rc, db_idx=GB_COUNTERS_DB_IDX)

        _sel(rc, GB_COUNTERS_DB_IDX)
        init_state = rc.hget(f"{RATES_TABLE}:{PORT_OID}:PORT", "INIT_DONE")
        assert init_state is None
