#!/bin/bash

# --- SETUP ---
SRC_FILE="src/main.cpp"
EXEC_NAME="spmv_exec"

# Carica moduli necessari per compilazione e qsub
module load mpich-3.2
# module load python-3.8.1 # (Opzionale qui, serve solo se lo usi per altro)

# --- FASE 0: Compilazione (Sempre) ---
echo "[LAUNCHER] Compilazione C++..."
mpic++ -O3 $SRC_FILE -o $EXEC_NAME
if [ $? -ne 0 ]; then
    echo "[ERROR] Compilazione fallita! Interrompo tutto."
    exit 1
fi
echo "[OK] Compilazione riuscita."

# --- LOGICA ARGOMENTI ---
MODE=$1  # Legge il primo argomento (bin, simulate, o vuoto)

# Funzione per lanciare la conversione
submit_bin() {
    echo "[LAUNCHER] Sottomissione Job Conversione..."
    # qsub restituisce l'ID del job (es: 12345.cluster_name)
    # Lo salviamo nella variabile CONVERT_JOB_ID
    CONVERT_JOB_ID=$(qsub convert_job.pbs)
    echo " -> Job Conversione inviato: $CONVERT_JOB_ID"
}

# Funzione per lanciare la simulazione
submit_sim() {
    DEP_ID=$1 # Argomento opzionale: ID del job da aspettare
    
    echo "[LAUNCHER] Sottomissione Job Simulazione..."
    
    if [ -z "$DEP_ID" ]; then
        # Nessuna dipendenza: lancia subito
        SIM_JOB_ID=$(qsub simulate_job.pbs)
    else
        # CON DIPENDENZA: Usa il flag -W depend=afterok:ID
        # Significa: "Parti solo dopo che ID è finito con successo (ok)"
        echo " -> In attesa che finisca il job: $DEP_ID"
        SIM_JOB_ID=$(qsub -W depend=afterok:$DEP_ID simulate_job.pbs)
    fi
    
    echo " -> Job Simulazione inviato: $SIM_JOB_ID"
}

# --- SELEZIONE MODALITÀ ---

if [ "$MODE" == "bin" ]; then
    # Caso 1: Solo conversione
    submit_bin

elif [ "$MODE" == "simulate" ]; then
    # Caso 2: Solo simulazione (presuppone binari già pronti)
    submit_sim

else
    # Caso 3 (Default): Tutto (Conversione -> Simulazione)
    echo "[LAUNCHER] Modalità FULL: Conversione seguita da Simulazione"
    
    # 1. Lancia conversione e cattura l'ID
    submit_bin
    
    # 2. Lancia simulazione passando l'ID della conversione
    submit_sim $CONVERT_JOB_ID
fi