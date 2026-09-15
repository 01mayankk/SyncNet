"""
SyncNet Reactive Decision State Machine Simulation Engine
=========================================================
Keeps neural detection (GNN embeddings + cluster coordination scoring) separate from response logic.

Simulated States:
- NORMAL    : Score < 0.35
- FLAGGED   : 0.35 <= Score < 0.60
- THROTTLED : 0.60 <= Score < 0.85
- ESCALATED : Score >= 0.85
- DECAYING  : Automatic exponential score reduction over time intervals
- APPEALED  : Manual override resetting score to 0.0 and state to NORMAL

Simulation Safety Notice:
All enforcement actions are purely simulated for research and decision support.
"""

import os
import math

class ReactiveStateMachine:
    def __init__(self, initial_score: float = 0.0, target_id: str = "cluster_00"):
        self.target_id = target_id
        self.initial_score = float(initial_score)
        self.current_score = float(initial_score)
        self.history = []
        self.state = self.evaluate_score(self.current_score)

    def evaluate_score(self, score: float) -> str:
        """Determines discrete simulation state based on score thresholds."""
        score = max(0.0, min(1.0, float(score)))
        self.current_score = score
        
        if score >= 0.85:
            new_state = "ESCALATED"
        elif score >= 0.60:
            new_state = "THROTTLED"
        elif score >= 0.35:
            new_state = "FLAGGED"
        else:
            new_state = "NORMAL"
            
        self.state = new_state
        self.history.append({"action": "evaluate", "score": score, "state": new_state})
        return new_state

    def step_decay(self, t_steps: int = 1, decay_rate: float = 0.05, min_threshold: float = 0.0) -> float:
        """
        Applies mathematical score decay over time intervals:
        S(t) = max(S_min, S_0 * e^(-lambda * t))
        """
        if t_steps < 0:
            raise ValueError("Time steps t_steps must be non-negative.")
            
        decayed_score = self.current_score * math.exp(-decay_rate * t_steps)
        decayed_score = max(min_threshold, max(0.0, min(1.0, decayed_score)))
        
        self.current_score = round(decayed_score, 4)
        
        # Determine updated state after decay step
        if self.current_score >= 0.85:
            new_state = "ESCALATED"
        elif self.current_score >= 0.60:
            new_state = "THROTTLED"
        elif self.current_score >= 0.35:
            new_state = "FLAGGED"
        else:
            new_state = "NORMAL"
            
        # Tag as DECAYING if score decreased but remains active
        if decayed_score < self.initial_score and new_state in ["FLAGGED", "THROTTLED"]:
            self.state = "DECAYING"
        else:
            self.state = new_state
            
        self.history.append({"action": f"decay_t{t_steps}", "score": self.current_score, "state": self.state})
        return self.current_score

    def submit_appeal(self, reviewer_reason: str = "Verified authentic network") -> str:
        """Simulates manual appeal review, resetting score to 0.0 and state to APPEALED then NORMAL."""
        self.history.append({"action": "appeal_submitted", "score": self.current_score, "state": "APPEALED", "reason": reviewer_reason})
        self.current_score = 0.0
        self.state = "APPEALED"
        return self.state

def run_reactive_simulation_demo():
    print("=" * 60)
    print("SYNCNET — REACTIVE DECISION STATE MACHINE SIMULATION")
    print("=" * 60)
    
    rsm = ReactiveStateMachine(initial_score=0.82, target_id="cluster_03")
    print(f"Initial Setup  : Score = {rsm.current_score:.4f} | State = {rsm.state}")
    
    # Step 1: Decay over 5 time intervals
    rsm.step_decay(t_steps=5)
    print(f"Decay t=5      : Score = {rsm.current_score:.4f} | State = {rsm.state}")
    
    # Step 2: Decay over 10 time intervals
    rsm.step_decay(t_steps=10)
    print(f"Decay t=10     : Score = {rsm.current_score:.4f} | State = {rsm.state}")
    
    # Step 3: Escalate score
    rsm.evaluate_score(0.92)
    print(f"Score Escalated: Score = {rsm.current_score:.4f} | State = {rsm.state}")
    
    # Step 4: Submit Appeal
    rsm.submit_appeal("Approved reviewer appeal")
    print(f"Appeal Reset   : Score = {rsm.current_score:.4f} | State = {rsm.state}")
    print("=" * 60)
    return rsm

if __name__ == "__main__":
    run_reactive_simulation_demo()
