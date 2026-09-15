"""
Pytest Unit Test Suite for SyncNet Reactive Decision State Machine
===================================================================
Tests state transitions, threshold triggers, mathematical score decay, and appeal resets.
"""

import math
import pytest
from scripts.reactive_simulator import ReactiveStateMachine

def test_initial_state_thresholds():
    rsm_norm = ReactiveStateMachine(initial_score=0.20)
    assert rsm_norm.state == "NORMAL"
    assert rsm_norm.current_score == 0.20

    rsm_flag = ReactiveStateMachine(initial_score=0.45)
    assert rsm_flag.state == "FLAGGED"

    rsm_throt = ReactiveStateMachine(initial_score=0.75)
    assert rsm_throt.state == "THROTTLED"

    rsm_escal = ReactiveStateMachine(initial_score=0.90)
    assert rsm_escal.state == "ESCALATED"

def test_score_evaluation_transitions():
    rsm = ReactiveStateMachine(initial_score=0.10)
    assert rsm.state == "NORMAL"

    rsm.evaluate_score(0.50)
    assert rsm.state == "FLAGGED"

    rsm.evaluate_score(0.70)
    assert rsm.state == "THROTTLED"

    rsm.evaluate_score(0.95)
    assert rsm.state == "ESCALATED"

    rsm.evaluate_score(0.0)
    assert rsm.state == "NORMAL"

def test_exponential_decay_math():
    initial_score = 0.80
    rsm = ReactiveStateMachine(initial_score=initial_score)
    assert rsm.state == "THROTTLED"

    # Step decay t=5, lambda=0.05
    # Expected: 0.80 * exp(-0.05 * 5) = 0.80 * exp(-0.25) = 0.80 * 0.77880078 = 0.6230
    decayed_score = rsm.step_decay(t_steps=5, decay_rate=0.05)
    expected_score = round(initial_score * math.exp(-0.25), 4)
    assert abs(decayed_score - expected_score) < 1e-3
    assert rsm.state in ["THROTTLED", "DECAYING"]

    # Step decay t=20 -> score should drop below 0.35 threshold into NORMAL
    rsm.step_decay(t_steps=20, decay_rate=0.05)
    assert rsm.current_score < 0.35
    assert rsm.state == "NORMAL"

def test_appeal_reset():
    rsm = ReactiveStateMachine(initial_score=0.88)
    assert rsm.state == "ESCALATED"

    rsm.submit_appeal(reviewer_reason="False positive cluster verified")
    assert rsm.state == "APPEALED"
    assert rsm.current_score == 0.0

def test_invalid_decay_time_steps():
    rsm = ReactiveStateMachine(initial_score=0.50)
    with pytest.raises(ValueError):
        rsm.step_decay(t_steps=-5)
