#!/usr/bin/env bash
#
# Mechanical checks for claims and links that are easy for prose edits to
# break. This is intentionally small. It catches objective drift and leaves
# questions of voice, pacing, and explanation to human review.
set -uo pipefail

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
failures=0
em_dash=$'\u2014'
MAX_DIAGRAM_COLUMNS=72
FENCE_PATTERN='^[[:space:]]*(```|~~~)'
MARKDOWN_LINK_PATTERN='\[([^][]*)\]\(([^()]*)\)'
SYMBOL_LABEL_PATTERN='^`([A-Za-z_][A-Za-z0-9_]*)`$'

choose_utf8_locale() {
    if LC_ALL=C.UTF-8 locale charmap 2>/dev/null | grep -qi '^UTF-8$'; then
        export LC_ALL=C.UTF-8
        return
    fi
    if LC_ALL=C.utf8 locale charmap 2>/dev/null | grep -qi '^UTF-8$'; then
        export LC_ALL=C.utf8
        return
    fi
    printf 'check-book: a UTF-8 locale is required to measure diagrams\n'
    exit 1
}

markdown_fragment_exists() {
    local path=$1
    local wanted=$2
    local heading
    local slug

    while IFS= read -r heading; do
        heading=${heading#* }
        slug=$(
            printf '%s' "$heading" \
                | LC_ALL=C tr '[:upper:]' '[:lower:]' \
                | sed -E 's/[`*_~]//g; s/[^a-z0-9 _-]//g; s/[[:space:]]+/-/g'
        )
        if [[ $slug == "$wanted" ]]; then
            return 0
        fi
    done < <(grep -E '^#{1,6}[[:space:]]' "$path" || true)
    return 1
}

check_diagram_layout() {
    local document=$1
    local line_number=0
    local box_open=0
    local box_column=0
    local box_width=0
    local text
    local prefix
    local remainder
    local content
    local suffix

    while IFS= read -r text || [[ -n $text ]]; do
        line_number=$((line_number + 1))

        if [[ $text == *$'\t'* ]]; then
            printf 'check-book: %s:%s: tab character can break diagram alignment\n' \
                "$document" "$line_number"
            failures=$((failures + 1))
        fi

        case "$text" in
            *'┌'*|*'┐'*|*'└'*|*'┘'*|*'│'*|*'├'*|*'┤'*|*'┬'*|*'┴'*|*'▼'*|*'▲'*|*'►'*)
                if ((${#text} > MAX_DIAGRAM_COLUMNS)); then
                    printf 'check-book: %s:%s: diagram line is %s columns; maximum is %s\n' \
                        "$document" "$line_number" "${#text}" \
                        "$MAX_DIAGRAM_COLUMNS"
                    failures=$((failures + 1))
                fi
                ;;
        esac

        if [[ $text == *'┌'*'┐'* ]]; then
            if ((box_open)); then
                printf 'check-book: %s:%s: box begins before the previous box closes\n' \
                    "$document" "$line_number"
                failures=$((failures + 1))
            fi

            prefix=${text%%┌*}
            remainder=${text#*┌}
            content=${remainder%%┐*}
            suffix=${remainder#*┐}

            if [[ $prefix =~ [^[:space:]] || $suffix =~ [^[:space:]] ]]; then
                printf 'check-book: %s:%s: box border shares a line with unrelated content\n' \
                    "$document" "$line_number"
                failures=$((failures + 1))
            fi

            box_column=${#prefix}
            box_width=${#content}
            box_open=1
            continue
        fi

        if ((box_open)); then
            if [[ $text == *'└'*'┘'* ]]; then
                prefix=${text%%└*}
                remainder=${text#*└}
                content=${remainder%%┘*}
                suffix=${remainder#*┘}

                if ((${#prefix} != box_column
                     || ${#content} != box_width)) \
                   || [[ $prefix =~ [^[:space:]]
                         || $suffix =~ [^[:space:]] ]]; then
                    printf 'check-book: %s:%s: closing border does not align with its box\n' \
                        "$document" "$line_number"
                    failures=$((failures + 1))
                fi
                box_open=0
            elif [[ $text == *'│'* && ${text#*│} == *'│'* ]]; then
                prefix=${text%%│*}
                remainder=${text#*│}
                content=${remainder%│*}
                suffix=${remainder##*│}

                if ((${#prefix} != box_column
                     || ${#content} != box_width)) \
                   || [[ $prefix =~ [^[:space:]]
                         || $suffix =~ [^[:space:]] ]]; then
                    printf 'check-book: %s:%s: content border does not align with its box\n' \
                        "$document" "$line_number"
                    failures=$((failures + 1))
                fi
            else
                printf 'check-book: %s:%s: box is interrupted before its closing border\n' \
                    "$document" "$line_number"
                failures=$((failures + 1))
                box_open=0
            fi
        fi
    done < "$document"

    if ((box_open)); then
        printf 'check-book: %s: box has no closing border\n' "$document"
        failures=$((failures + 1))
    fi
}

check_source_link() {
    local document=$1
    local line_number=$2
    local label=$3
    local target=$4
    local path=$5
    local document_directory=$6
    local symbol

    if [[ $target =~ \#L[0-9] ]]; then
        printf 'check-book: %s:%s: brittle source line anchor %s\n' \
            "$document" "$line_number" "$target"
        failures=$((failures + 1))
    fi
    if [[ ! $label =~ $SYMBOL_LABEL_PATTERN ]]; then
        return
    fi

    symbol=${BASH_REMATCH[1]}
    if grep -Eq "(^|[^A-Za-z0-9_])${symbol}([^A-Za-z0-9_]|$)" \
            "$document_directory/$path"; then
        return
    fi
    printf 'check-book: %s:%s: symbol %s is absent from %s\n' \
        "$document" "$line_number" "$symbol" "$path"
    failures=$((failures + 1))
}

check_line_fragment() {
    local document=$1
    local line_number=$2
    local path=$3
    local fragment=$4
    local document_directory=$5
    local first_line=$6
    local last_line=$7
    local target_lines

    target_lines=$(wc -l < "$document_directory/$path")
    if ((first_line >= 1 && last_line >= first_line
         && last_line <= target_lines)); then
        return
    fi
    printf 'check-book: %s:%s: invalid line anchor %s for %s-line file\n' \
        "$document" "$line_number" "$fragment" "$target_lines"
    failures=$((failures + 1))
}

check_document_link() {
    local document=$1
    local line_number=$2
    local label=$3
    local target=$4
    local document_directory
    local path
    local fragment
    local first_line
    local last_line

    if [[ $target == \<* ]]; then
        target=${target#<}
        target=${target%%>*}
    else
        target=${target%%[[:space:]]*}
    fi

    case "$target" in
        ""|\#*|//*) return ;;
    esac
    [[ $target =~ ^[A-Za-z][A-Za-z0-9+.-]*: ]] && return

    path=${target%%#*}
    path=${path%%\?*}
    path=${path//\\ / }
    [[ -n $path ]] || return

    document_directory=$(dirname -- "$document")
    if [[ ! -e "$document_directory/$path" ]]; then
        printf 'check-book: %s:%s: missing local link target %s\n' \
            "$document" "$line_number" "$target"
        failures=$((failures + 1))
        return
    fi
    if [[ $path == *.c || $path == *.h ]]; then
        check_source_link "$document" "$line_number" "$label" "$target" \
            "$path" "$document_directory"
        return
    fi
    [[ $target == *#* ]] || return

    fragment=${target#*#}
    if [[ $fragment =~ ^L([0-9]+)(-L([0-9]+))?$ ]]; then
        first_line=${BASH_REMATCH[1]}
        last_line=${BASH_REMATCH[3]:-${BASH_REMATCH[1]}}
        check_line_fragment "$document" "$line_number" "$path" \
            "$fragment" "$document_directory" "$first_line" "$last_line"
        return
    fi
    if [[ $path == *.md ]] \
       && ! markdown_fragment_exists "$document_directory/$path" "$fragment"; then
        printf 'check-book: %s:%s: missing heading fragment %s in %s\n' \
            "$document" "$line_number" "$fragment" "$path"
        failures=$((failures + 1))
    fi
}

check_document_links() {
    local document=$1
    local line_number=0
    local in_fence=0
    local fence_marker=
    local text
    local remainder
    local full_match
    local label
    local target

    while IFS= read -r text || [[ -n $text ]]; do
        line_number=$((line_number + 1))

        if ((in_fence)); then
            if [[ $text =~ ^[[:space:]]*$fence_marker ]]; then
                in_fence=0
                fence_marker=
            fi
            continue
        fi
        if [[ $text =~ $FENCE_PATTERN ]]; then
            in_fence=1
            fence_marker=${BASH_REMATCH[1]}
            continue
        fi

        remainder=$text
        while [[ $remainder =~ $MARKDOWN_LINK_PATTERN ]]; do
            full_match=${BASH_REMATCH[0]}
            label=${BASH_REMATCH[1]}
            target=${BASH_REMATCH[2]}
            check_document_link "$document" "$line_number" "$label" "$target"
            remainder=${remainder#*"$full_match"}
        done
    done < "$document"

    if ((in_fence)); then
        printf 'check-book: %s: unclosed Markdown code fence\n' "$document"
        failures=$((failures + 1))
    fi
}

main() {
    local matches
    local release_version
    local document
    local -a markdown_files
    local -a project_text_files

    choose_utf8_locale
    cd "$REPOSITORY_ROOT" || exit 1

    mapfile -d '' markdown_files < <(
        find . \
            -path './.git' -prune -o \
            -path './build' -prune -o \
            -type f -name '*.md' -print0
    )
    if matches=$(grep -nF -- "$em_dash" "${markdown_files[@]}"); then
        printf '%s\n' "check-book: Unicode U+2014 em dash found:" "$matches"
        failures=$((failures + 1))
    fi

    mapfile -d '' project_text_files < <(
        find README.md CHANGELOG.md CONTRIBUTING.md MODEL_CARD.md NOTICE \
             CITATION.cff VERSION .gitattributes .gitignore Makefile \
             data/LICENSE.md \
             book labs scripts src tests .github \
             -type f ! -path '*/build/*' \
             ! -path 'scripts/check-book.sh' -print0
    )
    if matches=$(
        grep -nEi -- 'tiny[ -]?grok|grok\.bin|TGRK' \
            "${project_text_files[@]}"
    ); then
        printf '%s\n' "check-book: retired project branding found:" "$matches"
        failures=$((failures + 1))
    fi

    release_version=$(tr -d '\r\n' < VERSION)
    if [[ ! $release_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
       || ! grep -qF "version: $release_version" CITATION.cff \
       || ! grep -qF "## $release_version - " CHANGELOG.md; then
        printf 'check-book: VERSION, CITATION.cff, and CHANGELOG.md disagree\n'
        failures=$((failures + 1))
    fi

    for document in "${markdown_files[@]}"; do
        check_diagram_layout "$document"
        check_document_links "$document"
    done

    if matches=$(
        grep -RniE --include='*.md' \
            "the checked-in result|checked-in corpus's|current checked-in corpus" \
            book
    ); then
        printf '%s\n' \
            "check-book: generated data is described as checked in:" \
            "$matches"
        failures=$((failures + 1))
    fi

    if ((failures > 0)); then
        printf 'check-book: %d check group(s) failed\n' "$failures"
        exit 1
    fi
    printf 'check-book: Markdown links and mechanical prose checks passed\n'
}

main "$@"
