#!/bin/bash

# A script to check for required dependencies for the Coral Protocol setup.
# This script only verifies dependencies and optionally installs uv if missing.

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u
# Ensure that pipeline commands fail on the first error.
set -o pipefail

# --- Color Definitions ---
C_BLUE='\033[0;34m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[1;33m'
C_RED='\033[0;31m'
C_NC='\033[0m' # No Color

# --- Main Function ---

# Function to compare version numbers
version_compare() {
    local version1=$1
    local version2=$2

    # Convert versions to comparable format (remove non-numeric characters except dots)
    version1=$(echo "$version1" | sed 's/[^0-9.]//g')
    version2=$(echo "$version2" | sed 's/[^0-9.]//g')

    # Cross-platform version comparison
    # First try sort -V (GNU coreutils), fallback to manual comparison
    if printf '%s\n%s\n' "$version2" "$version1" | sort -V -C 2>/dev/null; then
        return 0  # version1 >= version2
    elif command -v sort >/dev/null 2>&1 && sort --version-sort /dev/null 2>/dev/null; then
        # GNU sort with --version-sort
        if printf '%s\n%s\n' "$version2" "$version1" | sort --version-sort -C; then
            return 0
        else
            return 1
        fi
    else
        # Fallback: manual version comparison for macOS and other systems
        # Split versions into arrays and compare numerically
        IFS='.' read -ra ver1_parts <<< "$version1"
        IFS='.' read -ra ver2_parts <<< "$version2"

        # Pad arrays to same length
        local max_len=${#ver1_parts[@]}
        if [ ${#ver2_parts[@]} -gt $max_len ]; then
            max_len=${#ver2_parts[@]}
        fi

        for ((i=0; i<max_len; i++)); do
            local v1=${ver1_parts[i]:-0}
            local v2=${ver2_parts[i]:-0}

            if [ "$v1" -gt "$v2" ]; then
                return 0  # version1 > version2
            elif [ "$v1" -lt "$v2" ]; then
                return 1  # version1 < version2
            fi
        done

        return 0  # versions are equal
    fi
}

# Function to check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${C_RED}  [✗] Python 3 is not installed.${C_NC}"
        return 1
    fi

    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local required_version="3.10"

    if version_compare "$python_version" "$required_version"; then
        echo -e "${C_GREEN}  [✓] Python $python_version (required: $required_version+)${C_NC}"
        return 0
    else
        echo -e "${C_RED}  [✗] Python $python_version (required: $required_version+)${C_NC}"
        return 1
    fi
}

# Function to check Node.js version
check_node() {
    if ! command -v node &> /dev/null; then
        echo -e "${C_RED}  [✗] Node.js is not installed.${C_NC}"
        return 1
    fi

    local node_version=$(node --version 2>&1 | sed 's/v//')
    local required_version="18.0"

    if version_compare "$node_version" "$required_version"; then
        echo -e "${C_GREEN}  [✓] Node.js $node_version (required: 18+)${C_NC}"
        return 0
    else
        echo -e "${C_RED}  [✗] Node.js $node_version (required: 18+)${C_NC}"
        return 1
    fi
}

# Function to check npm (bundled with Node.js)
check_npm() {
    if ! command -v npm &> /dev/null; then
        echo -e "${C_RED}  [✗] npm is not installed.${C_NC}"
        return 1
    fi

    local npm_version=$(npm --version 2>&1)
    echo -e "${C_GREEN}  [✓] npm $npm_version (bundled with Node.js)${C_NC}"
    return 0
}

# Function to check Java version
check_java() {
    if ! command -v java &> /dev/null; then
        echo -e "${C_RED}  [✗] Java is not installed.${C_NC}"
        return 1
    fi

    # Get Java version (handle different output formats)
    local java_version_output=$(java -version 2>&1 | head -n 1)
    local java_version

    if [[ $java_version_output =~ \"([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        # Old format like "1.8.0_XXX"
        java_version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
        if [[ ${BASH_REMATCH[1]} == "1" ]]; then
            java_version="${BASH_REMATCH[2]}"
        fi
    elif [[ $java_version_output =~ \"([0-9]+)\.([0-9]+) ]]; then
        # Format like "17.0.X"
        java_version="${BASH_REMATCH[1]}"
    elif [[ $java_version_output =~ \"([0-9]+) ]]; then
        # Simple format like "21"
        java_version="${BASH_REMATCH[1]}"
    else
        echo -e "${C_YELLOW}  [!] Java version format not recognized: $java_version_output${C_NC}"
        return 0  # Assume it's okay if we can't parse it
    fi

    local required_version="21"

    if version_compare "$java_version" "$required_version"; then
        echo -e "${C_GREEN}  [✓] Java $java_version (required: $required_version+)${C_NC}"
        return 0
    else
        echo -e "${C_RED}  [✗] Java $java_version (required: $required_version+)${C_NC}"
        return 1
    fi
}

# Function to check Git (latest recommended)
check_git() {
    if ! command -v git &> /dev/null; then
        echo -e "${C_RED}  [✗] Git is not installed.${C_NC}"
        return 1
    fi

    local git_version=$(git --version 2>&1 | cut -d' ' -f3)
    echo -e "${C_GREEN}  [✓] Git $git_version (latest recommended)${C_NC}"
    return 0
}

# Function to check uv version
check_uv() {
    if ! command -v "uv" &> /dev/null; then
        echo -e "${C_YELLOW}  [!] uv is not installed.${C_NC}"
        echo -e "${C_YELLOW}Would you like to install uv automatically? (y/n): ${C_NC}"
        read -r response

        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo -e "${C_YELLOW}Installing uv...${C_NC}"
            curl -LsSf https://astral.sh/uv/install.sh | sh

            # Source the shell profile to make uv available in current session
            # Try different possible locations for cargo env
            if [ -f "$HOME/.cargo/env" ]; then
                source "$HOME/.cargo/env"
            elif [ -f "$HOME/.local/bin/uv" ]; then
                export PATH="$HOME/.local/bin:$PATH"
            fi

            # Verify installation
            if command -v "uv" &> /dev/null; then
                local uv_version=$(uv --version 2>&1 | cut -d' ' -f2)
                echo -e "${C_GREEN}  [✓] uv $uv_version has been successfully installed (latest)${C_NC}"
                return 0
            else
                echo -e "${C_RED}  [✗] Failed to install uv. Please install it manually.${C_NC}"
                return 1
            fi
        else
            echo -e "${C_RED}  [✗] uv is not installed. Please install it manually.${C_NC}"
            return 1
        fi
    else
        local uv_version=$(uv --version 2>&1 | cut -d' ' -f2)
        echo -e "${C_GREEN}  [✓] uv $uv_version (latest recommended)${C_NC}"
        return 0
    fi
}

## Check Dependencies
# Function to check for required dependencies with version requirements
check_dependencies() {
    echo -e "${C_BLUE}Checking for required dependencies and versions...${C_NC}"
    local missing_deps=0

    # Check each dependency with version requirements
    check_python || missing_deps=1
    check_uv || missing_deps=1
    check_node || missing_deps=1
    check_npm || missing_deps=1
    check_git || missing_deps=1
    check_java || missing_deps=1

    echo ""
    if [ "$missing_deps" -eq 1 ]; then
        echo -e "${C_RED}❌ Some dependencies are missing or don't meet version requirements.${C_NC}"
        echo -e "\n${C_BLUE}Installation/Upgrade instructions:${C_NC}"
        echo -e "  - ${C_YELLOW}Python 3.10+${C_NC}: Install from python.org or your package manager"
        echo -e "  - ${C_YELLOW}uv (latest)${C_NC}: Run 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        echo -e "  - ${C_YELLOW}Node.js 18+${C_NC}: Install from nodejs.org or your package manager"
        echo -e "  - ${C_YELLOW}npm${C_NC}: Bundled with Node.js (install Node.js 18+)"
        echo -e "  - ${C_YELLOW}Git (latest)${C_NC}: Install from git-scm.com or your package manager"
        echo -e "  - ${C_YELLOW}Java 21+${C_NC}: Install from your package manager or adoptium.net"
        echo -e "\n${C_BLUE}Additional requirements:${C_NC}"
        echo -e "  - ${C_YELLOW}OpenAI API Key${C_NC}: Required for agents (get from openai.com)"
        exit 1
    else
        echo -e "${C_GREEN}✅ All dependencies meet version requirements!${C_NC}"
        echo -e "\n${C_BLUE}Additional requirements to prepare:${C_NC}"
        echo -e "  - ${C_YELLOW}AIML API Key${C_NC}: Required for agents (get from aimlapi.com)"
        echo -e "  - ${C_YELLOW}ElevenLabs API Key${C_NC}: Optional for audio generation"
        echo -e "  - ${C_YELLOW}U.S. Census API Key${C_NC}: Optional for demographics data"
        echo -e "\n${C_GREEN}You can now run the full setup script: ./quickstart.sh${C_NC}"
    fi
}

# --- Script Execution ---
main() {
    echo -e "${C_BLUE}=====================================${C_NC}"
    echo -e "${C_BLUE}   Coral Protocol Dependency Check   ${C_NC}"
    echo -e "${C_BLUE}=====================================${C_NC}\n"

    check_dependencies
}

main