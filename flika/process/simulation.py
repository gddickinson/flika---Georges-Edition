# -*- coding: utf-8 -*-
"""Menu integration for microscopy simulation."""
import numpy as np
from flika.utils.BaseProcess import BaseProcess_noPriorWindow
import flika.global_vars as g

__all__ = ['simulate']


class Simulate(BaseProcess_noPriorWindow):
    """Open the Microscopy Simulation dialog or run a preset."""

    def __init__(self):
        super().__init__()

    def gui(self):
        """Open the Simulation Builder dialog."""
        from flika.simulation.dialog import SimulationDialog
        dlg = SimulationDialog(parent=g.m)
        dlg.show()
        if hasattr(g, 'dialogs'):
            g.dialogs.append(dlg)
        return dlg

    def run(self, preset='Beads - PSF Calibration', **overrides):
        """Run a simulation preset programmatically.

        Parameters
        ----------
        preset : str
            Name of the preset to run.
        **overrides
            Config attributes to override.

        Returns
        -------
        Window
            New flika Window containing the simulated data.
        """
        from flika.simulation.presets import PRESETS
        from flika.simulation.engine import SimulationEngine, SimulationConfig
        from dataclasses import asdict
        import copy

        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(PRESETS.keys())}")

        # Copy preset config and apply overrides
        base = PRESETS[preset]
        config = copy.deepcopy(base)
        for k, v in overrides.items():
            setattr(config, k, v)

        engine = SimulationEngine(config)
        stack, metadata = engine.run()

        from flika.window import Window
        w = Window(stack, name=f"Sim: {preset}")
        w.metadata['simulation'] = metadata
        return w

    def run_benchmarks_gui(self):
        """Open the Benchmark Runner dialog."""
        from flika.simulation.benchmark_dialog import BenchmarkDialog
        dlg = BenchmarkDialog(parent=g.m)
        dlg.show()
        if hasattr(g, 'dialogs'):
            g.dialogs.append(dlg)
        return dlg


simulate = Simulate()
