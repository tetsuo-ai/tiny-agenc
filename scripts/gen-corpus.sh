#!/usr/bin/env bash
#
# gen-corpus.sh -- synthesize the raw NIGHT GRID corpus with a local
# Ollama model.
#
# Each call to the model produces a few "transmissions": short dialogue
# scenes in a rigid format that a character-level model can learn.  The
# script appends to the output file until it reaches the target size, so
# it is safe to interrupt and re-run.  Pipe the result through
# clean-corpus.sh before training.
#
# Environment overrides: MODEL, RAW_FILE, TARGET_BYTES, OLLAMA_URL.

set -euo pipefail

MODEL="${MODEL:-dolphin3}"
RAW_FILE="${RAW_FILE:-data/cyberpunk.raw.txt}"
TARGET_BYTES="${TARGET_BYTES:-1200000}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/api/generate}"

TEMPERATURE=0.9         # high enough for variety, low enough to hold the format
MAX_TOKENS_PER_CALL=2048
CALL_TIMEOUT_SECONDS=600
SCENES_PER_CALL=3
PROGRESS_EVERY_CALLS=5
MAX_CONSECUTIVE_FAILURES=3

SPEAKERS="RAZR, GHOST, VYPR, NYX, JINX, DOC, WIRES, MOTH"
SLANG="choom, ice, mesh, jack in, stay frosty, chrome, corpo, netrun, blackout, cred-chips, the Grid"

SCENARIOS=(
    "a data heist inside a corpo tower after midnight"
    "a chase through the undercity on stolen bikes"
    "a deal gone wrong at the neon docks"
    "an extraction of a corpo defector from a rooftop pad"
    "a netrun into black ice guarding a vault of secrets"
    "a blackout swallowing an entire sector, and looters moving in"
    "an ambush at a noodle stall in the market sprawl"
    "a rogue AI whispering to the crew over the mesh"
    "smuggling military chrome through a checkpoint"
    "a mole inside the crew, and nobody knows who"
    "a ransom exchange on a mag-rail platform"
    "a wounded runner and a back-alley clinic with one bed left"
    "a memory broker selling someone the wrong past"
    "a bounty posted on one of the crew's own handles"
    "a dead drop that turns out to be a trap"
    "a ghost signal broadcasting from a tower nobody owns"
)

HEADER_LOCATIONS=(
    "BLACKOUT DISTRICT"
    "CHROME MARKET"
    "CORPORATE HEIGHTS"
    "DEAD SIGNAL WARD"
    "ICE HARBOR"
    "MAG-RAIL TERMINAL"
    "NEON UNDERCITY"
    "NIGHTGRID OUTSKIRTS"
    "OLD TOWNS"
    "ROOFTOP PAD"
    "SHADOW EXCHANGE"
    "SMOKESTACK DISTRICT"
)

HEADER_PATTERN="^=== TRANSMISSION [0-9]{4} // SECTOR ([1-9]|1[0-2]): [A-Z0-9][A-Z0-9 '-]* ===$"
DIALOGUE_PATTERN="^(RAZR|GHOST|VYPR|NYX|JINX|DOC|WIRES|MOTH): [^ ].*[.!?][\"']?$"

prompt_for() {
    local scenario="$1"
    local example_header="$2"
    cat <<EOF
You are generating training text for a tiny character-level language model.
Write ${SCENES_PER_CALL} scenes from the fictional cyberpunk serial NIGHT GRID
as transmission logs.

STRICT FORMAT, NO EXCEPTIONS:
- Each scene starts with a header line exactly like:
  ${example_header}
  (4-digit number, sector 1-12, location name in capitals)
- The example is syntax only. Invent a different header for every scene.
- After the header: 20 to 35 lines of dialogue. Each line is exactly
  HANDLE: one or two short sentences.
- Use ONLY these handles: ${SPEAKERS}. Use 2 to 4 of them per scene.
- Plain ASCII only. No emoji, no markdown, no asterisks, no stage
  directions, no narration, no text outside the format above.
- Sprinkle this slang naturally: ${SLANG}.
- Hard-boiled tone. Short punchy sentences. Never break character.

Scenario for these scenes: ${scenario}.
Output the transmissions and nothing else.
EOF
}

generate_once() {
    local scenario="$1"
    local example_header="$2"
    jq -n --arg model "$MODEL" \
          --arg prompt "$(prompt_for "$scenario" "$example_header")" \
          --argjson temp "$TEMPERATURE" \
          --argjson tokens "$MAX_TOKENS_PER_CALL" \
          '{model: $model, prompt: $prompt, stream: false,
            options: {temperature: $temp, num_predict: $tokens}}' \
    | timeout "$CALL_TIMEOUT_SECONDS" curl -s "$OLLAMA_URL" -d @- \
    | jq -r '.response // empty'
}

file_bytes() {
    if [ -f "$RAW_FILE" ]; then stat -c%s "$RAW_FILE"; else echo 0; fi
}

response_has_scene() {
    local response="$1"

    grep -qE "$HEADER_PATTERN" <<< "$response" \
        && grep -qE "$DIALOGUE_PATTERN" <<< "$response"
}

mkdir -p "$(dirname "$RAW_FILE")"

calls=0
failures=0
while [ "$(file_bytes)" -lt "$TARGET_BYTES" ]; do
    scenario="${SCENARIOS[$((calls % ${#SCENARIOS[@]}))]}"
    calls=$((calls + 1))

    header_number="$(shuf -i 0-9999 -n 1)"
    printf -v header_number '%04d' "$header_number"
    header_sector="$(shuf -i 1-12 -n 1)"
    header_location="${HEADER_LOCATIONS[$(shuf -i 0-$((${#HEADER_LOCATIONS[@]} - 1)) -n 1)]}"
    example_header="=== TRANSMISSION $header_number // SECTOR $header_sector: $header_location ==="

    response="$(generate_once "$scenario" "$example_header")" || response=""
    if [ -z "$response" ] || ! response_has_scene "$response"; then
        failures=$((failures + 1))
        if [ "$failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo "gen-corpus: $failures unusable responses in a row;" \
                 "is Ollama up and serving $MODEL?" >&2
            exit 1
        fi
        continue
    fi
    failures=0

    printf '%s\n\n' "$response" >> "$RAW_FILE"
    if [ $((calls % PROGRESS_EVERY_CALLS)) -eq 0 ]; then
        echo "gen-corpus: $(file_bytes) / $TARGET_BYTES bytes after $calls calls"
    fi
done

echo "gen-corpus: done, $(file_bytes) bytes in $RAW_FILE after $calls calls"
