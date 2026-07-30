# Tiny AgenC -- a character-level autoregressive transformer in plain C.
#
#   make            build ./tiny-agenc
#   make check      run gradient checks and teaching integration tests
#   make check-foundations/data/mat run focused chapter checks
#   make check-forward/backward/optimizer run focused math checks
#   make check-model run the model contract and full-model math checks
#   make check-sampling run controlled temperature and context witnesses
#   make smoke      run the tiny train, save, load, and sample smoke test
#   make check-cli  exercise the public CLI and scene-level data split
#   make overfit    prove the assembled learner can memorize one batch
#   make bigram-baseline compare held-out loss with a count model
#   make check-book check local links and objective prose rules
#   make check-evidence verify the evidence manifest and recorded metrics
#   make check-metadata verify release-version metadata stays synchronized
#   make check-install stage and exercise the installed program and model
#   make check-checkpoint exercise durable-save metadata and failure paths
#   make check-sanitizers run core checks under AddressSanitizer and UBSan
#   make check-ndebug compile and test the model with assertions disabled
#   make corpus     clean the committed raw NIGHT GRID corpus without network access
#   make generate-corpus extend raw data with Ollama, then clean it
#   make validation-data split the cleaned corpus at scene boundaries
#   make data       fetch and verify the pinned Shakespeare comparison corpus
#   make install    install the program, model, licenses, and artifact metadata
#   make clean      remove build products while preserving build/migration
#
# OpenMP is on by default; build single-threaded with `make OPENMP=0`.

# -ffast-math lets the compiler reassociate reductions and apply more
# aggressive floating-point optimizations.  The measured tradeoff is
# recorded in book/logs/dev-measurements.md. The portable default does
# not tune for the build host. `make NATIVE=1` adds -march=native; the
# recorded 5,000-step evidence used that setting.
CC       ?= cc
OPENMP   ?= 1
NATIVE   ?= 0
OPTFLAGS ?= -O3 -ffast-math
CFLAGS += -std=c11 $(OPTFLAGS) -Wall -Wextra -Werror
LDLIBS += -lacl -lm
PREFIX   ?= /usr/local
BINDIR   ?= $(PREFIX)/bin
DATADIR  ?= $(PREFIX)/share
DOCDIR   ?= $(DATADIR)/doc/tiny-agenc
MODELDIR ?= $(DATADIR)/tiny-agenc
DESTDIR  ?=

ifeq ($(OPENMP),1)
CFLAGS += -fopenmp
else ifeq ($(OPENMP),0)
CFLAGS += -Wno-unknown-pragmas
else
$(error OPENMP must be 0 or 1)
endif

ifeq ($(NATIVE),1)
CFLAGS += -march=native
else ifneq ($(NATIVE),0)
$(error NATIVE must be 0 or 1)
endif

SRC     := src/util.c src/rng.c src/tokenizer.c src/dataset.c src/ops.c \
           src/param.c src/model.c src/model_parameters.c src/model_memory.c \
           src/model_forward.c src/model_backward.c src/model_sampling.c \
           src/checkpoint.c
BUILD   ?= build/openmp-$(OPENMP)
OBJ     := $(SRC:src/%.c=$(BUILD)/%.o)
HEADERS := $(wildcard src/*.h)
BUILD_CONFIG := $(BUILD)/.build-config
INSTALL_DOCS := CHANGELOG.md CITATION.cff EVIDENCE.md EVIDENCE.sha256 \
                LICENSE MODEL_CARD.md NOTICE README.md VERSION

# The public target intentionally relinks. Mode-specific directories and
# the configuration stamp below prevent stale objects when the compiler,
# flags, OpenMP mode, or native-code setting changes.
tiny-agenc: $(BUILD)/main.o $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/gradcheck: $(BUILD)/gradcheck.o $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/integration: $(BUILD)/integration.o $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/overfit: $(BUILD)/overfit.o $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/bigram: $(BUILD)/bigram.o $(BUILD)/util.o $(BUILD)/rng.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/sampling: $(BUILD)/sampling.o $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/model-invalid: $(BUILD)/model-invalid.o $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

CHECKPOINT_WRAP_FLAGS := \
	-Wl,--wrap=acl_cmp \
	-Wl,--wrap=acl_set_fd \
	-Wl,--wrap=close \
	-Wl,--wrap=fchmod \
	-Wl,--wrap=fchown \
	-Wl,--wrap=fclose \
	-Wl,--wrap=fflush \
	-Wl,--wrap=fmemopen \
	-Wl,--wrap=fsetxattr \
	-Wl,--wrap=fsync \
	-Wl,--wrap=renameat \
	-Wl,--wrap=renameat2

$(BUILD)/checkpoint-failures: $(BUILD)/checkpoint-failures.o $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $(CHECKPOINT_WRAP_FLAGS) \
		-o $@ $^ $(LDLIBS)

build/split-order: scripts/split-order.c src/rng.c src/util.c src/rng.h \
                   src/util.h $(BUILD_CONFIG)
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -Wno-unknown-pragmas -Isrc \
		-o $@ scripts/split-order.c src/rng.c src/util.c $(LDLIBS)

$(BUILD)/%.o: src/%.c $(HEADERS) $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(BUILD)/gradcheck.o: tests/gradcheck.c $(HEADERS) $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -c -o $@ $<

$(BUILD)/integration.o: tests/integration.c $(HEADERS) $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -c -o $@ $<

$(BUILD)/overfit.o: tests/overfit.c $(HEADERS) $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -c -o $@ $<

$(BUILD)/bigram.o: tests/bigram.c $(HEADERS) $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -c -o $@ $<

$(BUILD)/sampling.o: tests/sampling.c $(HEADERS) $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -c -o $@ $<

$(BUILD)/model-invalid.o: tests/model_invalid.c $(HEADERS) $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -c -o $@ $<

$(BUILD)/checkpoint-failures.o: tests/checkpoint_failures.c $(HEADERS) \
                                $(BUILD_CONFIG) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -c -o $@ $<

$(BUILD):
	mkdir -p $@

$(BUILD_CONFIG): FORCE | $(BUILD)
	@temporary="$@.tmp.$$$$"; \
	trap 'rm -f -- "$$temporary"' EXIT HUP INT TERM; \
	{ \
		printf '%s\n' \
			"CC=$(CC)" \
			"CPPFLAGS=$(CPPFLAGS)" \
			"CFLAGS=$(CFLAGS)" \
			"LDFLAGS=$(LDFLAGS)" \
			"LDLIBS=$(LDLIBS)" \
			"OPENMP=$(OPENMP)" \
			"NATIVE=$(NATIVE)"; \
		$(CC) --version 2>/dev/null | sed -n '1p'; \
	} > "$$temporary"; \
	if test -r "$@" && cmp -s "$$temporary" "$@"; then \
		rm -f -- "$$temporary"; \
	else \
		mv -f -- "$$temporary" "$@"; \
	fi; \
	trap - EXIT HUP INT TERM

FORCE:

check-gradient: $(BUILD)/gradcheck
	./$(BUILD)/gradcheck

check-integration: $(BUILD)/integration
	./$(BUILD)/integration

check: check-gradient check-integration check-checkpoint

check-checkpoint: $(BUILD)/checkpoint-failures
	./$(BUILD)/checkpoint-failures

check-foundations: $(BUILD)/integration
	./$(BUILD)/integration foundations

check-data: $(BUILD)/integration
	./$(BUILD)/integration data

check-mat: $(BUILD)/integration
	./$(BUILD)/integration mat

check-forward: $(BUILD)/integration
	./$(BUILD)/integration forward

check-parallel: $(BUILD)/integration
	./$(BUILD)/integration parallel

check-backward: $(BUILD)/gradcheck
	./$(BUILD)/gradcheck backward

check-optimizer: $(BUILD)/gradcheck $(BUILD)/integration
	./$(BUILD)/gradcheck optimizer
	./$(BUILD)/integration optimizer

check-model: $(BUILD)/gradcheck $(BUILD)/integration
	./$(BUILD)/gradcheck model
	./$(BUILD)/integration model

check-sampling: $(BUILD)/sampling
	./$(BUILD)/sampling

check-cli: tiny-agenc build/split-order
	bash scripts/smoke-cli.sh ./tiny-agenc

smoke: check-cli check-sampling $(BUILD)/integration
	./$(BUILD)/integration smoke

overfit: $(BUILD)/overfit
	./$(BUILD)/overfit

bigram-baseline: validation-data $(BUILD)/bigram
	./$(BUILD)/bigram data/cyberpunk.train.txt data/cyberpunk.val.txt

check-book:
	bash scripts/check-book.sh

check-labs:
	env CC="$(CC)" OPENMP="$(OPENMP)" bash scripts/check-labs.sh

check-evidence: validation-data tiny-agenc $(BUILD)/bigram
	bash scripts/check-evidence.sh ./tiny-agenc ./$(BUILD)/bigram

check-metadata:
	bash scripts/check-release-metadata.sh

check-install: tiny-agenc tiny-agenc.bin $(INSTALL_DOCS)
	@set -eu; \
	staging=$$(mktemp -d); \
	trap 'rm -rf -- "$$staging"' EXIT HUP INT TERM; \
	$(MAKE) --no-print-directory DESTDIR="$$staging" PREFIX=/usr install; \
	bash scripts/check-install.sh "$$staging/usr"

check-sanitizers:
	$(MAKE) OPENMP=0 NATIVE=0 BUILD=build/sanitizers \
		OPTFLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined" \
		LDFLAGS="-fsanitize=address,undefined" \
		check check-sampling overfit

check-invalid-construction: $(BUILD)/model-invalid
	./$(BUILD)/model-invalid

check-ndebug:
	$(MAKE) --no-print-directory OPENMP=0 NATIVE=0 BUILD=build/ndebug \
		CPPFLAGS="$(CPPFLAGS) -DNDEBUG" \
		check-model check-invalid-construction

check-all: check check-parallel check-cli check-sampling overfit \
           check-book check-labs check-evidence check-metadata check-ndebug
	$(MAKE) --no-print-directory check-install

corpus:
	bash scripts/clean-corpus.sh

generate-corpus:
	bash scripts/gen-corpus.sh
	bash scripts/clean-corpus.sh

validation-data: corpus build/split-order
	bash scripts/split-corpus.sh

data:
	bash scripts/get-shakespeare.sh

install: tiny-agenc tiny-agenc.bin $(INSTALL_DOCS)
	install -d "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(DOCDIR)" \
		"$(DESTDIR)$(MODELDIR)"
	install -m 0755 tiny-agenc "$(DESTDIR)$(BINDIR)/tiny-agenc"
	install -m 0644 $(INSTALL_DOCS) "$(DESTDIR)$(DOCDIR)"
	install -m 0644 tiny-agenc.bin "$(DESTDIR)$(MODELDIR)/tiny-agenc.bin"

clean:
	@if test -d build; then \
		find build -mindepth 1 -maxdepth 1 ! -name migration \
			-exec rm -rf -- {} +; \
	fi
	rm -f tiny-agenc

.PHONY: tiny-agenc $(BUILD)/gradcheck $(BUILD)/integration $(BUILD)/overfit \
        $(BUILD)/bigram $(BUILD)/sampling $(BUILD)/model-invalid \
        $(BUILD)/checkpoint-failures \
        build/split-order \
        check-gradient \
        check-integration check-checkpoint check check-foundations check-data check-mat \
        check-parallel \
        check-forward check-backward check-optimizer check-model \
        check-sampling check-cli \
        smoke overfit \
        bigram-baseline \
        check-book check-labs check-evidence check-metadata check-install \
        check-sanitizers check-invalid-construction check-ndebug check-all \
        corpus generate-corpus validation-data data install clean FORCE
