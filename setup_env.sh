#!/bin/bash
set -e

# CodeXRAG Installation Script
# ============================
# Sets up the environment for the Scientific Research Assistant.

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CodeXRAG 2.0 Installer ===${NC}"

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}[+] Python found: $(python3 --version)${NC}"

# 2. Setup Virtual Environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE} Creating virtual environment in $VENV_DIR...${NC}"
    python3 -m venv "$VENV_DIR"
else
    echo -e "${GREEN}[+] Virtual environment exists.${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# 3. Install Dependencies
echo -e "${BLUE} Installing dependencies via pip...${NC}"
pip install --upgrade pip
pip install -e .[rerank]

# 4. Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Warning: 'ollama' command not found.${NC}"
    echo "Please install Ollama from https://ollama.com/ to use local models."
else
    echo -e "${GREEN}[+] Ollama found.${NC}"
    
    # Check if server is running
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo -e "${GREEN}[+] Ollama server is running.${NC}"
        
        # Pull models (if not present)
        echo -e "${BLUE} Checking models...${NC}"
        ollama pull qwen2.5-coder:14b
    else
        echo -e "${RED}Warning: Ollama server is NOT running. Start it with 'ollama serve'.${NC}"
    fi
fi

echo -e "${GREEN}=== Installation Complete ===${NC}"
echo "To activate: source .venv/bin/activate"
echo "To run CLI:    codexrag chat --repo ."
echo "To run Web UI: streamlit run codexrag/gui.py"
