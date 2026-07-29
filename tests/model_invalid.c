/*
 * A tiny death witness for model_new's public configuration boundary.
 * The dimensions remain cheap to construct if that boundary disappears.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "model.h"

#ifndef NDEBUG
#error "model_invalid.c must be compiled with NDEBUG"
#endif

static void construct_invalid_model(void)
{
    ModelConfig invalid = {
        .vocab_size  = 2,
        .block_size  = 2,
        .d_model     = 3,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = 1,
    };

    Model *model = model_new(invalid, 17);

    model_free(model);
}

int main(void)
{
    pid_t child = fork();

    if (child < 0) {
        perror("check-invalid-construction: fork");
        return EXIT_FAILURE;
    }
    if (child == 0) {
        const struct rlimit no_core = { 0, 0 };

        (void)setrlimit(RLIMIT_CORE, &no_core);
        (void)close(STDERR_FILENO);
        construct_invalid_model();
        _exit(EXIT_SUCCESS);
    }

    int status;

    if (waitpid(child, &status, 0) < 0) {
        perror("check-invalid-construction: waitpid");
        return EXIT_FAILURE;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_FAILURE) {
        fputs("check-invalid-construction: model_new did not reject "
              "invalid geometry with EXIT_FAILURE\n", stderr);
        return EXIT_FAILURE;
    }
    puts("check-invalid-construction: model_new rejected invalid geometry");
    return EXIT_SUCCESS;
}
