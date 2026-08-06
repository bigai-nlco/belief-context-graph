#!/bin/sh
set -eu

REPOSITORY="${BCG_REPOSITORY:-bigai-nlco/belief-context-graph}"
VERSION="${BCG_VERSION:-main}"

fail() {
    printf 'bcg installer: %s\n' "$1" >&2
    exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required."
command -v tar >/dev/null 2>&1 || fail "tar is required."
command -v node >/dev/null 2>&1 || fail "Node.js >=22.19 is required."
command -v npm >/dev/null 2>&1 || fail "npm is required."

node_version="$(node -p 'process.versions.node')"
node_major="$(printf '%s' "$node_version" | cut -d. -f1)"
node_minor="$(printf '%s' "$node_version" | cut -d. -f2)"
if [ "$node_major" -lt 22 ] || {
    [ "$node_major" -eq 22 ] && [ "$node_minor" -lt 19 ]
}; then
    fail "Node.js >=22.19 is required; found $node_version."
fi

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export PATH
fi
command -v uv >/dev/null 2>&1 || fail "uv installation did not add uv to PATH."

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

archive="$temporary_dir/bcg.tar.gz"
source_url="https://codeload.github.com/$REPOSITORY/tar.gz/$VERSION"
printf 'Downloading BCG %s...\n' "$VERSION"
curl -LsSf "$source_url" -o "$archive"
tar -xzf "$archive" -C "$temporary_dir"

source_dir=""
for candidate in "$temporary_dir"/*; do
    if [ -d "$candidate/agent-cli" ] && [ -f "$candidate/pyproject.toml" ]; then
        source_dir="$candidate"
        break
    fi
done
[ -n "$source_dir" ] || fail "Downloaded archive does not contain a BCG source package."

printf '%s\n' "Installing the Python launcher and Graph Construction runtime..."
uv tool install --python 3.11 --force --refresh-package bcg "$source_dir"

printf '%s\n' "Building and installing the terminal Agent..."
npm --prefix "$source_dir/agent-cli" ci
npm --prefix "$source_dir/agent-cli" run build
npm --prefix "$source_dir/agent-cli" pack --pack-destination="$temporary_dir" --loglevel=error >/dev/null
agent_archive=""
for candidate in "$temporary_dir"/*.tgz; do
    if [ -f "$candidate" ]; then
        agent_archive="$candidate"
        break
    fi
done
[ -n "$agent_archive" ] || fail "Could not package the terminal Agent."
if ! npm install --global "$agent_archive" --no-audit --no-fund; then
    printf '%s\n' "bcg-agent installation failed. bcg may already be installed;"
    printf 'rerun this installer or run %s to install only the Agent.\n' "make install-tool" >&2
    exit 1
fi

if command -v bcg >/dev/null 2>&1 && command -v bcg-agent >/dev/null 2>&1; then
    printf '\n%s\n' "BCG is installed. Run:"
    printf '  %s\n' "bcg"
else
    bin_dir="$(uv tool dir --bin 2>/dev/null || true)"
    npm_bin="$(npm prefix --global 2>/dev/null)/bin"
    printf '\n%s\n' "BCG is installed, but one or more tool directories are not on PATH."
    if ! command -v bcg >/dev/null 2>&1; then
        printf 'Run %s and restart your shell.\n' "'uv tool update-shell'"
        if [ -n "$bin_dir" ]; then
            printf 'uv tool bin directory: %s\n' "$bin_dir"
        fi
    fi
    if ! command -v bcg-agent >/dev/null 2>&1; then
        printf 'Add the npm global bin directory to PATH: %s\n' "$npm_bin"
    fi
    printf 'Then run %s.\n' "'bcg'"
fi

printf '%s\n' "The first launch opens a guided setup and stores configuration in ~/.bcg/."
