Simple & Secure Terminal Commands

Basic Commands:

# Full seeding (reset + seed + scheduler)
python worker.py

# Just seeding without scheduler
python worker.py seeding

# Reset seeding status only
python worker.py reset

# Force reset + seeding
python worker.py seeding --force

# FMP only
python worker.py fmp

# FUNDATA only
python worker.py fundata

Module-style execution:

python -m worker seeding
python -m worker fmp
python -m worker reset

Python one-liners:

python -c "import worker; worker.force_fresh_seeding(); worker.initial_seeding()"
python -c "import worker; worker.fmp_seeding()"