#include <stdio.h>
#include <stdlib.h>

#include "spec.h"

static int failures;

static void expect(int condition, const char *message);
int main(void);

static void expect(int condition, const char *message)
{
    if (condition)
        return;
    fprintf(stderr, "check-01: %s\n", message);
    failures++;
}

int main(void)
{
    expect(lab_config_valid(80, 128, 128, 4, 4, 32),
           "the default configuration must be valid");
    expect(!lab_config_valid(80, 128, 127, 4, 4, 32),
           "width must divide evenly among heads");
    expect(!lab_config_valid(0, 128, 128, 4, 4, 32),
           "vocabulary size must be positive");
    expect(!lab_config_valid(MODEL_MAX_VOCAB_SIZE + 1,
                             128, 128, 4, 4, 32),
           "byte vocabulary cannot exceed 256 entries");
    expect(!lab_config_valid(80, MODEL_MAX_BLOCK_SIZE + 1,
                             128, 4, 4, 1),
           "context must stay within the documented ceiling");
    expect(!lab_config_valid(80, 128, MODEL_MAX_D_MODEL + 1,
                             4, 4, 1),
           "width must stay within the documented ceiling");
    expect(!lab_config_valid(80, 128, 128,
                             MODEL_MAX_HEAD_COUNT + 1, 4, 1),
           "head count must stay within the documented ceiling");
    expect(!lab_config_valid(80, 128, 128, 4,
                             MODEL_MAX_LAYER_COUNT + 1, 1),
           "layer count must stay within the documented ceiling");
    expect(lab_config_valid(80, MODEL_MAX_BLOCK_SIZE, 128, 4, 1,
                            MODEL_MAX_TOKENS_PER_PASS
                            / MODEL_MAX_BLOCK_SIZE),
           "the token-count ceiling is inclusive");
    expect(!lab_config_valid(80, MODEL_MAX_BLOCK_SIZE, 128, 4, 1,
                             MODEL_MAX_TOKENS_PER_PASS
                             / MODEL_MAX_BLOCK_SIZE + 1),
           "batch times context cannot exceed the token ceiling");
    expect(lab_parameter_count(80, 128, 128, 4) == 815360,
           "default parameter count must be 815360");

    size_t large_expected =
        ((size_t)65536 + (size_t)65536) * (size_t)16384
        + (size_t)(12 * 16384 + 4) * (size_t)16384
        + (size_t)2 * (size_t)16384;

    expect(lab_parameter_count(65536, 65536, 16384, 1) == large_expected,
           "parameter counting must not overflow int");

    if (failures != 0)
        return EXIT_FAILURE;
    puts("check-01: model geometry passed");
    return EXIT_SUCCESS;
}
