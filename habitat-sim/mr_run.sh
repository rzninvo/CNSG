#!/bin/bash

# --- Attivazione conda environment ---
# Carica il sistema di conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Controlla se l'env attivo è habitat
if [[ "$CONDA_DEFAULT_ENV" != "habitat" ]]; then
    echo "Activating conda environment: habitat"
    conda activate habitat
else
    echo "Conda environment 'habitat' already active"
fi

# --- Avvio dei processi ---
python3 examples/mr_viewer.py &
PYTHON_PID=$!


echo "Python PID: $PYTHON_PID"

# Funzione che viene eseguita quando premi CTRL+C o il processo viene killato
cleanup() {
    echo ""
    echo "Closing processes..."

    kill $PYTHON_PID 2>/dev/null

    echo "Processes terminated."
    exit 0
}

# Cattura Ctrl+C (SIGINT) e kill (SIGTERM)
trap cleanup SIGINT SIGTERM

# Attende che il programma Python termini
wait $PYTHON_PID

cleanup
