#!/bin/bash

# Fine-Tuning Repo Verification Script
# Checks for professional standards (LOC, Structure, Required Files)

RED='\033[0-1;31m'
GREEN='\033[0-1;32m'
BLUE='\033[0-1;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Fine-Tuning Repo Verification ===${NC}"

# 1. Check for Required Files
echo -en "Checking for required files... "
REQUIRED_FILES=("LICENSE" "NOTICE" "README.md" "labels.yml" ".gitignore" ".beads/config.yaml" ".agents/steering/dev-principles.md" "scripts/verify_repo.sh")
MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Missing $file${NC}"
        MISSING=1
    fi
done

if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

# 2. Check for Role Definitions
echo -en "Checking for Kiro agent roles... "
if [ "$(ls .kiro/agents/*.md 2>/dev/null | wc -l)" -ge 6 ]; then
    echo -e "${GREEN}OK (6+ roles)${NC}"
else
    echo -e "${RED}FAILED (Missing roles)${NC}"
fi

# 3. Line Count Check (< 200 LOC)
echo -en "Checking for < 200 LOC per file... "
FILES_OVER_LOC=$(find . -maxdepth 4 -name "*.py" -o -name "*.md" | xargs wc -l | awk '$1 > 200 && $2 != "total" {print $2 " (" $1 " lines)"}')
if [ -z "$FILES_OVER_LOC" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo -e "$FILES_OVER_LOC"
fi

# 4. Git Status
echo -en "Checking git status... "
if [ -d ".git" ]; then
    echo -e "${GREEN}OK (Git initialized)${NC}"
else
    echo -e "${RED}FAILED (Git not found)${NC}"
fi

echo -e "${BLUE}=== Verification Complete ===${NC}"
