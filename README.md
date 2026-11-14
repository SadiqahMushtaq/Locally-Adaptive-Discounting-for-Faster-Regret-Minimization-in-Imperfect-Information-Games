# CFR+ project (Multiagent Systems)

Baseline for: "Can predictive/adaptive updates speed up CFR+ on small imperfect-information games?"

## Files

- `baseline_test.py`: smoke test (Kuhn poker, CFR+). Run this first.
- `run_baseline.py`: CFR vs CFR+ on Kuhn and Leduc; writes `results/`.
- `ablation.py`: toggles each CFR+ switch (alternating / linear / RM+); writes `results/ablation_<game>.*`.
- `notes/cfr_reading_guide.md`: annotated map of OpenSpiel's `cfr.py` (line numbers for open_spiel 2.0.2).
- `requirements.txt`

## One-time setup on Windows (WSL2 + VS Code)

OpenSpiel's pip package targets Linux/macOS, so use WSL2.

1. PowerShell (as admin): `wsl --install`, restart, finish Ubuntu setup.
2. Install VS Code, then the extension "WSL" (by Microsoft) and "Python".
3. In the Ubuntu terminal:

   ```bash
   sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
   python3 --version      # use 3.10 - 3.12 if possible
   ```

4. Make the virtual environment in your Linux home, NOT inside the OneDrive folder
   (OneDrive would try to sync thousands of venv files):

   ```bash
   mkdir -p ~/venvs && python3 -m venv ~/venvs/cfr
   source ~/venvs/cfr/bin/activate
   pip install -r "/mnt/c/Users/Administrator/OneDrive - Oregon State University/Q1/Multiagent Systems/Project/cfr-project/requirements.txt"
   ```

   If `pip install open_spiel` says "no matching distribution", your Python is probably too
   new. Install Python 3.11 or 3.12 (e.g. `sudo apt install python3.12 python3.12-venv`, or
   use `uv venv --python 3.12`) and recreate the venv.

5. Open the project in VS Code from Ubuntu:

   ```bash
   cd "/mnt/c/Users/Administrator/OneDrive - Oregon State University/Q1/Multiagent Systems/Project/cfr-project"
   code .
   ```

   Then in VS Code: Ctrl+Shift+P, "Python: Select Interpreter", choose `~/venvs/cfr/bin/python`.

## Run

```bash
python baseline_test.py     # should print exploitability shrinking toward 0
python run_baseline.py      # writes results/baseline_results.csv and .png
python ablation.py          # Kuhn by default; --game leduc_poker --iters 300 for Leduc
```

If Leduc is slow, lower it: `python run_baseline.py --iters-leduc 300`.

## Expected outcome

On a log-log plot, CFR+ should reach lower exploitability than CFR at the same iteration
count, on both games. If it does not, something is off; fix that before modifying anything.
