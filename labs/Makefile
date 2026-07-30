# Staged build-along checks. WORK may be absolute or relative to this file.
CC      ?= cc
WORK    ?= work
OPENMP  ?= 0
CFLAGS  ?= -std=c11 -O0 -g -Wall -Wextra -Werror -Wno-unknown-pragmas
LDLIBS  ?= -lacl -lm

ifeq ($(OPENMP),1)
CFLAGS += -fopenmp
else ifneq ($(OPENMP),0)
$(error OPENMP must be 0 or 1)
endif

ROOT     := ..
REF      := $(ROOT)/src
TESTS    := $(ROOT)/tests
BUILD    := build

INTEGRATION := $(TESTS)/integration.c
GRADCHECK   := $(TESTS)/gradcheck.c
OVERFIT     := $(TESTS)/overfit.c
MODEL_BASE  := "$(WORK)/model.c" "$(WORK)/model_parameters.c"
MODEL_CORE  := $(MODEL_BASE) "$(WORK)/model_memory.c"
MODEL_MATH  := $(MODEL_CORE) "$(WORK)/model_forward.c" \
               "$(WORK)/model_backward.c"
MODEL_IMPL  := $(MODEL_MATH) "$(WORK)/model_sampling.c" \
               "$(WORK)/checkpoint.c"
REF_MODEL   := $(REF)/model.c $(REF)/model_parameters.c \
               $(REF)/model_memory.c $(REF)/model_forward.c \
               $(REF)/model_backward.c $(REF)/model_sampling.c \
               $(REF)/checkpoint.c

$(BUILD):
	mkdir -p $@

check-01: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -o $(BUILD)/check-01 \
		check01.c "$(WORK)/spec.c"
	./$(BUILD)/check-01

check-02: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -o $(BUILD)/check-02 \
		check02.c "$(WORK)/util.c" "$(WORK)/rng.c" $(LDLIBS)
	./$(BUILD)/check-02

check-03: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -o $(BUILD)/check-03 \
		check03.c "$(WORK)/util.c" "$(WORK)/rng.c" \
		"$(WORK)/tokenizer.c" "$(WORK)/dataset.c" $(LDLIBS)
	./$(BUILD)/check-03

check-04: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -o $(BUILD)/check-04 check04.c
	./$(BUILD)/check-04

check-05: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -o $(BUILD)/check-05 \
		check05.c "$(WORK)/ops.c" $(LDLIBS)
	./$(BUILD)/check-05

check-06: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-06 \
		$(GRADCHECK) \
		$(REF)/util.c $(REF)/rng.c $(REF)/tokenizer.c \
		$(REF)/dataset.c "$(WORK)/ops.c" $(REF)/param.c \
		$(REF_MODEL) $(LDLIBS)
	./$(BUILD)/check-06 backward

check-parallel:
	$(MAKE) OPENMP=1 check-parallel-openmp

check-parallel-openmp: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) \
		-o $(BUILD)/check-parallel \
		$(INTEGRATION) \
		$(REF)/util.c $(REF)/rng.c $(REF)/tokenizer.c \
		$(REF)/dataset.c "$(WORK)/ops.c" $(REF)/param.c \
		$(REF_MODEL) $(LDLIBS)
	./$(BUILD)/check-parallel parallel

check-08: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-08 \
		check08.c "$(WORK)/param.c" \
		$(REF)/rng.c $(REF)/util.c $(LDLIBS)
	./$(BUILD)/check-08

check-blueprint:
	@cmp -s model_internal.h.in "$(REF)/model_internal.h" || { \
		printf '%s\n' \
		  'check-blueprint: lab scaffold differs from the reference private layout' >&2; \
		exit 1; \
	}
	@cmp -s model_internal.h.in "$(WORK)/model_internal.h" || { \
		printf '%s\n' \
		  'check-blueprint: restore the supplied model_internal.h without changes' >&2; \
		exit 1; \
	}
	@printf 'check-blueprint: the shared private layout is exact\n'

check-09: check-blueprint | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-09 \
		check09.c \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/param.c" \
		$(MODEL_BASE) $(LDLIBS)
	./$(BUILD)/check-09

check-10: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-10 \
		check10.c \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/ops.c" \
		"$(WORK)/param.c" $(MODEL_CORE) \
		$(REF)/model_forward.c $(REF)/model_backward.c $(LDLIBS)
	./$(BUILD)/check-10

check-11: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-11 \
		check11.c \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/ops.c" \
		"$(WORK)/param.c" $(MODEL_CORE) "$(WORK)/model_forward.c" \
		$(LDLIBS)
	./$(BUILD)/check-11

check-12: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-12 \
		check12.c \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/ops.c" \
		"$(WORK)/param.c" $(MODEL_MATH) $(LDLIBS)
	./$(BUILD)/check-12
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-12-gradients \
		$(GRADCHECK) \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/dataset.c" "$(WORK)/ops.c" "$(WORK)/param.c" \
		$(MODEL_MATH) $(LDLIBS)
	./$(BUILD)/check-12-gradients model

check-13: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-13 \
		check13.c \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/param.c" $(MODEL_CORE) "$(WORK)/checkpoint.c" \
		$(LDLIBS)
	./$(BUILD)/check-13

check-14: tiny-agenc-before-sampling
	bash check14.sh ./$(BUILD)/tiny-agenc-before-sampling

check-15: tiny-agenc-before-sampling
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-15-model \
		check15-model.c \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/dataset.c" "$(WORK)/ops.c" "$(WORK)/param.c" \
		$(MODEL_MATH) "$(WORK)/checkpoint.c" $(LDLIBS)
	./$(BUILD)/check-15-model $(BUILD)/check-15.bin
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-overfit \
		$(OVERFIT) \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/dataset.c" "$(WORK)/ops.c" "$(WORK)/param.c" \
		$(MODEL_MATH) $(LDLIBS)
	./$(BUILD)/check-overfit
	./$(BUILD)/tiny-agenc-before-sampling train --data tiny-corpus.txt \
		--out $(BUILD)/step-1.bin --steps 1 \
		--layers 1 --heads 1 --width 8 \
		--block 8 --batch 1 --seed 1337
	./$(BUILD)/tiny-agenc-before-sampling train --data tiny-corpus.txt \
		--out $(BUILD)/step-50.bin --steps 50 \
		--layers 1 --heads 1 --width 8 \
		--block 8 --batch 1 --seed 1337 > $(BUILD)/train-50.log
	! cmp -s $(BUILD)/step-1.bin $(BUILD)/step-50.bin
	awk '/^step/{if (!seen++) first=$$5; last=$$5} \
	     END{exit !(seen >= 2 && last < first)}' $(BUILD)/train-50.log
	./$(BUILD)/tiny-agenc-before-sampling train --data tiny-corpus.txt \
		--out $(BUILD)/without-validation.bin --steps 2 \
		--layers 1 --heads 1 --width 8 \
		--block 8 --batch 1 --seed 1337 > /dev/null
	cp tiny-corpus.txt $(BUILD)/validation-corpus.txt
	./$(BUILD)/tiny-agenc-before-sampling train --data tiny-corpus.txt \
		--val-data $(BUILD)/validation-corpus.txt \
		--out $(BUILD)/with-validation.bin \
		--steps 2 --layers 1 --heads 1 --width 8 \
		--block 8 --batch 1 --seed 1337 > /dev/null
	cmp $(BUILD)/without-validation.bin $(BUILD)/with-validation.bin

check-16: check-15 tiny-agenc
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-16-sampling \
		check16.c \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/dataset.c" "$(WORK)/ops.c" "$(WORK)/param.c" \
		$(MODEL_IMPL) $(LDLIBS)
	./$(BUILD)/check-16-sampling
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/check-16 \
		$(INTEGRATION) \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/dataset.c" "$(WORK)/ops.c" "$(WORK)/param.c" \
		$(MODEL_IMPL) $(LDLIBS)
	./$(BUILD)/check-16 smoke
	./$(BUILD)/tiny-agenc sample --model $(BUILD)/step-50.bin \
		--prompt RAZR: --length 24 --seed 1337 > $(BUILD)/sample-a.txt
	./$(BUILD)/tiny-agenc sample --model $(BUILD)/step-50.bin \
		--prompt RAZR: --length 24 --seed 1337 > $(BUILD)/sample-b.txt
	cmp $(BUILD)/sample-a.txt $(BUILD)/sample-b.txt
	./$(BUILD)/tiny-agenc sample --model $(BUILD)/step-50.bin \
		--prompt RAZR:@ --length 24 --seed 1337 > $(BUILD)/sample-unknown.txt
	cmp $(BUILD)/sample-a.txt $(BUILD)/sample-unknown.txt

tiny-agenc-before-sampling: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) \
		-o $(BUILD)/tiny-agenc-before-sampling \
		"$(WORK)/main.c" \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/dataset.c" "$(WORK)/ops.c" "$(WORK)/param.c" \
		$(MODEL_MATH) "$(WORK)/checkpoint.c" \
		$(REF)/model_sampling.c $(LDLIBS)

tiny-agenc: | $(BUILD)
	$(CC) $(CFLAGS) -I"$(WORK)" -I$(REF) -o $(BUILD)/tiny-agenc \
		"$(WORK)/main.c" \
		"$(WORK)/util.c" "$(WORK)/rng.c" "$(WORK)/tokenizer.c" \
		"$(WORK)/dataset.c" "$(WORK)/ops.c" "$(WORK)/param.c" \
		$(MODEL_IMPL) $(LDLIBS)

clean:
	rm -rf $(BUILD)

.PHONY: check-01 check-02 check-03 check-04 check-05 check-06 \
        check-parallel check-parallel-openmp check-08 check-09 check-10 \
        check-blueprint check-11 check-12 check-13 check-14 check-15 check-16 \
        tiny-agenc-before-sampling tiny-agenc clean
