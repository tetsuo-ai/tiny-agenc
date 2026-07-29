#!/usr/bin/env bash
#
# Keep the human-readable and compiled release versions synchronized.
set -euo pipefail

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPOSITORY_ROOT"
export LC_ALL=C

fail() {
    printf 'check-metadata: %s\n' "$*" >&2
    exit 1
}

mapfile -t version_lines < VERSION
[[ ${#version_lines[@]} == 1 ]] ||
    fail "VERSION must contain exactly one line"
version=${version_lines[0]}
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    fail "VERSION is not a three-part semantic version: $version"

citation_version=$(
    awk '$1 == "version:" {
        value = $2
        gsub(/^"/, "", value)
        gsub(/"$/, "", value)
        print value
        exit
    }' CITATION.cff
)
[[ $citation_version == "$version" ]] ||
    fail "CITATION.cff says $citation_version, VERSION says $version"

header_version=$(
    awk '$1 == "#define" && $2 == "TINY_AGENC_VERSION" {
        value = $3
        gsub(/"/, "", value)
        print value
        exit
    }' src/version.h
)
[[ $header_version == "$version" ]] ||
    fail "src/version.h says $header_version, VERSION says $version"

changelog_heading=$(
    grep -E "^## ${version//./\\.} - [0-9]{4}-[0-9]{2}-[0-9]{2}$" \
        CHANGELOG.md || true
)
[[ -n $changelog_heading ]] ||
    fail "CHANGELOG.md has no dated $version release heading"

changelog_date=${changelog_heading##* - }
citation_date=$(
    awk '$1 == "date-released:" { print $2; exit }' CITATION.cff
)
[[ $citation_date == "$changelog_date" ]] ||
    fail "release dates disagree: CITATION.cff has $citation_date, CHANGELOG.md has $changelog_date"

grep -qF "| Artifact version | $version |" MODEL_CARD.md ||
    fail "MODEL_CARD.md does not identify artifact version $version"

printf 'check-metadata: VERSION, CLI header, citation, changelog, and model card agree on %s\n' \
    "$version"
