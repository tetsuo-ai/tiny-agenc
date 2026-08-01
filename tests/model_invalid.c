/*
 * A tiny death witness for model_new's public configuration boundary.
 * The dimensions remain cheap to construct if that boundary disappears.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "model.h"

#ifndef NDEBUG
#error "model_invalid.c must be compiled with NDEBUG"
#endif

enum {
    PIPE_READ_END,
    PIPE_WRITE_END,
    DIAGNOSTIC_CAPACITY = 128,
};
static const unsigned long long MODEL_SEED = 17;

static void construct_model(ModelConfig config);
static int read_diagnostic(int descriptor, char *text, size_t capacity);
static int construction_is_rejected(ModelConfig config);
int main(void);

static void construct_model(ModelConfig config)
{
    Model *model = model_new(config, MODEL_SEED);

    model_free(model);
}

static int read_diagnostic(int descriptor, char *text, size_t capacity)
{
    char discard[DIAGNOSTIC_CAPACITY];
    size_t used = 0;
    int fits = capacity > 0;

    for (;;) {
        char *destination =
            used + 1 < capacity ? text + used : discard;
        size_t available =
            used + 1 < capacity ? capacity - used - 1 : sizeof discard;
        ssize_t received = read(descriptor, destination, available);

        if (received > 0) {
            if (destination == discard)
                fits = 0;
            else
                used += (size_t)received;
            continue;
        }
        if (received == 0)
            break;
        if (errno == EINTR)
            continue;
        fits = 0;
        break;
    }
    if (capacity > 0)
        text[used] = '\0';
    return fits;
}

static int construction_is_rejected(ModelConfig config)
{
    int diagnostic_pipe[2];

    if (pipe(diagnostic_pipe) != 0) {
        perror("check-invalid-construction: pipe");
        return 0;
    }
    pid_t child = fork();

    if (child < 0) {
        perror("check-invalid-construction: fork");
        (void)close(diagnostic_pipe[PIPE_READ_END]);
        (void)close(diagnostic_pipe[PIPE_WRITE_END]);
        return 0;
    }
    if (child == 0) {
        const struct rlimit no_core = { 0, 0 };

        (void)close(diagnostic_pipe[PIPE_READ_END]);
        (void)setrlimit(RLIMIT_CORE, &no_core);
        if (dup2(diagnostic_pipe[PIPE_WRITE_END], STDERR_FILENO) < 0)
            _exit(EXIT_FAILURE);
        (void)close(diagnostic_pipe[PIPE_WRITE_END]);
        construct_model(config);
        _exit(EXIT_SUCCESS);
    }

    (void)close(diagnostic_pipe[PIPE_WRITE_END]);
    char diagnostic[DIAGNOSTIC_CAPACITY];
    int diagnostic_ok =
        read_diagnostic(diagnostic_pipe[PIPE_READ_END], diagnostic,
                        sizeof diagnostic);

    (void)close(diagnostic_pipe[PIPE_READ_END]);
    int status;

    if (waitpid(child, &status, 0) < 0) {
        perror("check-invalid-construction: waitpid");
        return 0;
    }
    return diagnostic_ok
        && strstr(diagnostic,
                  "invalid or unrepresentable model configuration") != NULL
        && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE;
}

int main(void)
{
    ModelConfig uneven_heads = {
        .vocab_size  = 2,
        .block_size  = 2,
        .d_model     = 3,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = 1,
    };
    ModelConfig excessive_vocabulary = uneven_heads;
    ModelConfig excessive_pass = uneven_heads;

    excessive_vocabulary.vocab_size = MODEL_MAX_VOCAB_SIZE + 1;
    excessive_vocabulary.d_model = 2;
    excessive_pass.d_model = 2;
    excessive_pass.block_size = MODEL_MAX_BLOCK_SIZE;
    excessive_pass.batch_size =
        MODEL_MAX_TOKENS_PER_PASS / MODEL_MAX_BLOCK_SIZE + 1;

    if (!construction_is_rejected(uneven_heads)
        || !construction_is_rejected(excessive_vocabulary)
        || !construction_is_rejected(excessive_pass)) {
        fputs("check-invalid-construction: model_new did not reject an "
              "invalid configuration with EXIT_FAILURE\n", stderr);
        return EXIT_FAILURE;
    }
    puts("check-invalid-construction: model_new rejected invalid configurations");
    return EXIT_SUCCESS;
}
