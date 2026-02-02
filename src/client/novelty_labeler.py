# src/client/novelty_labeler.py
"""
Novelty detection and interactive labeling system.

Provides:
    - LabelMemory: Persistent storage for object labels and signatures
    - NoveltyController: Manages when to prompt user for labels
"""

import json
import os
import time
import sys
import select
from typing import Dict, Optional, Tuple, Any

import numpy as np

from .obs_processing import cosine_sim


class LabelMemory:
    """
    Persistent storage for object labels and visual signatures.
    
    Stores label-signature pairs and matches new observations
    against known prototypes using cosine similarity.
    """
    
    def __init__(
        self, 
        path: str = "labels.json", 
        sim_thresh: float = 0.90
    ):
        """
        Initialize label memory.
        
        Args:
            path: Path to JSON file for persistence
            sim_thresh: Similarity threshold for matching
        """
        self.path = path
        self.sim_thresh = sim_thresh
        self.data = {"objects": [], "actions": []}
        
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.data = json.load(f)
                print(f"[LabelMemory] Loaded {len(self.data['objects'])} labels from {path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[LabelMemory] Warning: Could not load {path}: {e}")
    
    def save(self) -> None:
        """Save labels to disk."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def match_object(
        self, 
        signature: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching label for signature.
        
        Args:
            signature: Visual signature vector
            
        Returns:
            Tuple of (label, similarity) or (None, best_sim) if no match
        """
        best = None
        best_sim = -1.0
        
        for item in self.data["objects"]:
            proto = np.array(item["signature"], dtype=np.float32)
            s = cosine_sim(proto, signature)
            if s > best_sim:
                best_sim = s
                best = item
        
        if best and best_sim >= self.sim_thresh:
            return best["label"], best_sim
        
        return None, best_sim
    
    def add_object(self, label: str, signature: np.ndarray) -> None:
        """
        Add new labeled object to memory.
        
        Args:
            label: Human-provided label
            signature: Visual signature vector
        """
        self.data["objects"].append({
            "label": label,
            "signature": signature.tolist(),
            "created_at": time.time()
        })
        self.save()
        print(f"[LabelMemory] Added label: '{label}'")
    
    def get_all_labels(self) -> list:
        """Get list of all known labels."""
        return [item["label"] for item in self.data["objects"]]
    
    def clear(self) -> None:
        """Clear all stored labels."""
        self.data = {"objects": [], "actions": []}
        self.save()


class NoveltyController:
    """
    Controls when to prompt user for object labels.
    
    Only prompts after:
        - Object has been visible for stable_steps consecutive frames
        - Object hasn't been asked about before
        - Signature doesn't match any known object
    """
    
    def __init__(
        self, 
        memory: LabelMemory, 
        stable_steps: int = 10,
        prompt_timeout: int = 30,
        auto_label_prefix: str = "unknown"
    ):
        """
        Initialize novelty controller.
        
        Args:
            memory: LabelMemory instance for storing labels
            stable_steps: Frames object must appear before prompting
            prompt_timeout: Seconds to wait for user input (0 = no timeout)
            auto_label_prefix: Prefix for auto-generated labels
        """
        self.mem = memory
        self.stable_steps = stable_steps
        self.prompt_timeout = prompt_timeout
        self.auto_label_prefix = auto_label_prefix
        
        self._seen_counts: Dict[int, int] = {}  # oid -> consecutive frames seen
        self._asked: set = set()  # oids we've already asked about
        self._pending_prompts: Dict[int, np.ndarray] = {}  # oid -> signature
    
    def update(
        self, 
        sigs_by_oid: Dict[int, np.ndarray]
    ) -> Dict[int, Optional[str]]:
        """
        Update tracking and return current labels for objects.
        
        Does NOT prompt user - call prompt_for_unknowns() separately.
        
        Args:
            sigs_by_oid: Dict mapping object_id -> signature vector
            
        Returns:
            Dict mapping object_id -> label (or None if unknown)
        """
        labels = {}
        
        # Track which objects are currently visible
        current_oids = set(sigs_by_oid.keys())
        
        for oid, sig in sigs_by_oid.items():
            # Update seen count
            self._seen_counts[oid] = self._seen_counts.get(oid, 0) + 1
            
            # Try to match against known objects
            label, sim = self.mem.match_object(sig)
            
            if label:
                labels[oid] = label
            else:
                labels[oid] = None
                
                # Mark for potential prompting if stable enough
                if (self._seen_counts[oid] >= self.stable_steps and 
                    oid not in self._asked):
                    self._pending_prompts[oid] = sig
        
        # Decay counts for objects that disappeared
        for oid in list(self._seen_counts.keys()):
            if oid not in current_oids:
                self._seen_counts[oid] = max(0, self._seen_counts[oid] - 1)
        
        return labels
    
    def has_pending_prompts(self) -> bool:
        """Check if there are objects waiting for labels."""
        return len(self._pending_prompts) > 0
    
    def get_pending_count(self) -> int:
        """Get number of objects waiting for labels."""
        return len(self._pending_prompts)
    
    def prompt_for_unknowns(self, interactive: bool = True) -> Dict[int, str]:
        """
        Prompt user for labels on unknown objects.
        
        Args:
            interactive: If True, prompt via terminal. If False, auto-label.
            
        Returns:
            Dict mapping object_id -> assigned label
        """
        new_labels = {}
        
        for oid, sig in list(self._pending_prompts.items()):
            self._asked.add(oid)
            
            if interactive:
                label = self._prompt_user(oid)
            else:
                label = f"{self.auto_label_prefix}_{int(time.time())}"
            
            if label:
                self.mem.add_object(label, sig)
                new_labels[oid] = label
            
            del self._pending_prompts[oid]
        
        return new_labels
    
    def _prompt_user(self, oid: int) -> Optional[str]:
        """
        Prompt user for label via terminal.
        
        Args:
            oid: Object ID to label
            
        Returns:
            User-provided label or auto-generated if timeout
        """
        print(f"\n{'='*50}")
        print(f"[Labeling] New object detected (id={oid})")
        print(f"[Labeling] Please provide a label for this object.")
        
        if self.prompt_timeout > 0:
            print(f"[Labeling] (Timeout in {self.prompt_timeout}s, will auto-label)")
            
            # Non-blocking input with timeout
            label = self._input_with_timeout(
                "Enter label (or blank to skip): ",
                self.prompt_timeout
            )
        else:
            label = input("Enter label (or blank to skip): ").strip()
        
        if not label:
            label = f"{self.auto_label_prefix}_{oid}_{int(time.time())}"
            print(f"[Labeling] Auto-assigned: '{label}'")
        
        print(f"{'='*50}\n")
        return label
    
    def _input_with_timeout(self, prompt: str, timeout: int) -> str:
        """
        Get user input with timeout.
        
        Args:
            prompt: Prompt string
            timeout: Timeout in seconds
            
        Returns:
            User input or empty string if timeout
        """
        print(prompt, end='', flush=True)
        
        # Platform-specific timeout handling
        if sys.platform == 'win32':
            # Windows doesn't support select on stdin
            import threading
            result = [None]
            
            def get_input():
                try:
                    result[0] = input()
                except EOFError:
                    result[0] = ""
            
            thread = threading.Thread(target=get_input)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if result[0] is None:
                print("\n[Labeling] Timeout reached")
                return ""
            return result[0].strip()
        else:
            # Unix-like systems
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                return sys.stdin.readline().strip()
            else:
                print("\n[Labeling] Timeout reached")
                return ""
    
    def reset(self) -> None:
        """Reset tracking state (but keep memory)."""
        self._seen_counts.clear()
        self._asked.clear()
        self._pending_prompts.clear()
