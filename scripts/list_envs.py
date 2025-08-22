# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.
"""

from isaaclab.app import AppLauncher
import gymnasium as gym
from prettytable import PrettyTable

# launch Omniverse headless
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ensure custom envs get registered
import ur5_isaaclab.tasks.manager_based.lift  # runs __init__.py and registers environments


def main():
    """Print all environments starting with 'Isaac-'."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for spec in gym.registry.values():
        if spec.id.startswith("Isaac-"):
            index += 1
            cfg = spec.kwargs.get("env_cfg_entry_point", "")
            table.add_row([index, spec.id, spec.entry_point, cfg])

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
