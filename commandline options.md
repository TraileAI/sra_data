Simple & Secure Terminal Commands

## Production Server Commands (Recommended for Render.com):

# Start server with health check + background worker (DEFAULT)
python server.py

# Start server with forced fresh seeding
python server.py --force

# Start server without background worker (health check only)
python server.py --no-scheduler

## Worker-Only Commands (No server):

# Just seeding without server
python server.py seeding

# Reset seeding status only
python server.py reset

# Force reset + seeding
python server.py seeding --force

# FMP only
python server.py fmp

# FUNDATA only
python server.py fundata

## Legacy Worker Commands (Still supported):

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

## Module-style execution:

python -m worker seeding
python -m worker fmp
python -m worker reset

## Python one-liners:

python -c "import worker; worker.force_fresh_seeding(); worker.initial_seeding()"
python -c "import worker; worker.fmp_seeding()"