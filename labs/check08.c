/*
 * Chapter 8 witness: AdamW under glass, then gradient tools.
 */
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "param.h"

static int checks;
static int failures;

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-08: %s\n", message);
    failures++;
}

static int close_float(float actual, float expected, float tolerance)
{
    return fabsf(actual - expected) <= tolerance;
}

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & 0x7F800000u) != 0x7F800000u;
}

static void run_reference(Param *p, AdamW opt, int steps, double decay,
                          const char *message)
{
    size_t count = mat_size(param_values(p));
    double *shadow = calloc(3 * count, sizeof *shadow);

    if (shadow == NULL) {
        expect(0, "reference allocation succeeds");
        return;
    }

    double *first = shadow + count;
    double *second = shadow + 2 * count;
    Mat values = param_values(p);

    for (size_t i = 0; i < count; i++)
        shadow[i] = values.vals[i];

    for (int step = 1; step <= steps; step++) {
        Mat gradient = param_gradient(p);

        for (size_t i = 0; i < count; i++) {
            float sign = ((int)i + step) % 2 == 0 ? 1.0f : -1.0f;

            gradient.vals[i] = sign * (float)(i + 1) * (0.05f * step);
        }

        for (size_t i = 0; i < count; i++) {
            double g = gradient.vals[i];

            first[i] = opt.beta1 * first[i] + (1.0 - opt.beta1) * g;
            second[i] = opt.beta2 * second[i] + (1.0 - opt.beta2) * g * g;

            double corrected_first =
                first[i] / (1.0 - pow(opt.beta1, step));
            double corrected_second =
                second[i] / (1.0 - pow(opt.beta2, step));

            shadow[i] -= opt.learning_rate
                       * (corrected_first
                          / (sqrt(corrected_second) + opt.epsilon)
                          + decay * shadow[i]);
        }
        expect(param_adamw_step(p, opt, step) == 0,
               "a valid AdamW step reports success");
    }

    values = param_values(p);
    for (size_t i = 0; i < count; i++)
        expect(close_float(values.vals[i], (float)shadow[i], 2e-5f), message);
    free(shadow);
}

static void check_adamw(void)
{
    AdamW opt = {
        .learning_rate = 0.1f,
        .beta1         = 0.9f,
        .beta2         = 0.999f,
        .epsilon       = 1e-8f,
        .weight_decay  = 0.01f,
    };
    Param *matrix = param_new_constant(2, 3, 0.75f);
    Param *vector = param_new_constant(1, 3, 0.75f);

    run_reference(matrix, opt, 4, opt.weight_decay,
                  "matrix agrees with the double-precision AdamW reference");
    run_reference(vector, opt, 4, 0.0,
                  "vector agrees with Adam without weight decay");

    param_zero_gradient(vector);
    expect(param_adamw_step(vector, opt, 5) == 0,
           "a valid zero-gradient step reports success");
    Mat values = param_values(vector);

    for (size_t i = 0; i < mat_size(values); i++)
        expect(finite_float(values.vals[i]),
               "a zero-gradient vector update stays finite");

    param_free(matrix);
    param_free(vector);
}

static void check_gradient_tools(void)
{
    Param *param = param_new_constant(2, 2, 0.0f);
    Mat gradient = param_gradient(param);

    gradient.vals[0] = 3.0f;
    gradient.vals[1] = 4.0f;
    gradient.vals[2] = -5.0f;
    gradient.vals[3] = 0.0f;

    param_scale_gradient(param, 0.25f);

    expect(close_float(gradient.vals[0], 0.75f, 1e-6f),
           "gradient scaling changes the first component");
    expect(close_float(gradient.vals[1], 1.0f, 1e-6f),
           "gradient scaling changes the second component");
    expect(close_float(gradient.vals[0] / gradient.vals[1],
                       3.0f / 4.0f, 1e-6f),
           "uniform gradient scaling preserves direction");
    expect(close_float(param_gradient_norm_squared(param), 3.125f, 1e-6f),
           "gradient norm reports every scaled element");

    param_zero_gradient(param);
    expect(param_gradient_norm_squared(param) == 0.0f,
           "gradient zeroing clears every element");
    param_free(param);
}

static void check_parameter_io(void)
{
    Param *original = param_new_constant(2, 2, 0.0f);
    Mat original_values = param_values(original);

    original_values.vals[0] = -3.5f;
    original_values.vals[1] = 0.0f;
    original_values.vals[2] = 1.25f;
    original_values.vals[3] = 99.0f;

    FILE *stream = tmpfile();

    expect(stream != NULL, "the parameter-I/O fixture opens");
    if (stream != NULL) {
        expect(param_write(original, stream) == 0,
               "finite parameter values write successfully");
        rewind(stream);

        Param *loaded = param_new_constant(2, 2, -1.0f);

        expect(param_read(loaded, stream) == 0,
               "a complete parameter payload reads successfully");
        Mat loaded_values = param_values(loaded);

        expect(memcmp(original_values.vals, loaded_values.vals,
                      mat_size(original_values) * sizeof *original_values.vals)
                   == 0,
               "parameter values round-trip bit for bit");
        expect(param_read(loaded, stream) != 0,
               "parameter reading rejects a short payload");
        param_free(loaded);
        fclose(stream);
    }

    uint32_t nan_bits = 0x7FC00000u;

    memcpy(&original_values.vals[0], &nan_bits, sizeof nan_bits);
    stream = tmpfile();
    expect(stream != NULL, "the non-finite parameter fixture opens");
    if (stream != NULL) {
        expect(param_write(original, stream) != 0,
               "parameter writing rejects non-finite values");
        fclose(stream);
    }
    param_free(original);
}

static void check_rejection_contracts(void)
{
    Rng *rng = rng_new(1);

    expect(param_new_constant(0, 2, 0.0f) == NULL,
           "a non-positive constant shape is rejected");
    expect(param_new_gaussian(2, 0, 0.02f, rng) == NULL,
           "a non-positive Gaussian shape is rejected");
    rng_free(rng);

    Param *tested = param_new_constant(1, 2, 0.5f);
    Param *control = param_new_constant(1, 2, 0.5f);
    Mat tested_gradient = param_gradient(tested);
    Mat control_gradient = param_gradient(control);

    tested_gradient.vals[0] = control_gradient.vals[0] = 0.25f;
    tested_gradient.vals[1] = control_gradient.vals[1] = -0.5f;

    AdamW valid = {
        .learning_rate = 0.01f,
        .beta1         = 0.9f,
        .beta2         = 0.999f,
        .epsilon       = 1e-8f,
        .weight_decay  = 0.01f,
    };
    AdamW invalid = valid;
    invalid.epsilon = 0.0f;

    expect(param_adamw_step(tested, invalid, 1) != 0,
           "an invalid AdamW recipe is rejected");
    expect(param_adamw_step(tested, valid, 1) == 0
           && param_adamw_step(control, valid, 1) == 0,
           "valid updates still succeed after a rejected recipe");

    Mat tested_values = param_values(tested);
    Mat control_values = param_values(control);

    expect(memcmp(tested_values.vals, control_values.vals,
                  mat_size(tested_values) * sizeof *tested_values.vals) == 0,
           "a rejected recipe leaves values and optimizer history unchanged");

    uint32_t infinity_bits = 0x7F800000u;

    memcpy(&tested_gradient.vals[0], &infinity_bits, sizeof infinity_bits);
    expect(!param_gradient_is_finite(tested),
           "the gradient scan identifies a non-finite element");
    expect(param_adamw_step(tested, valid, 2) != 0,
           "AdamW rejects a non-finite gradient");

    Param *unsafe = param_new_constant(2, 2, 0.5f);
    Param *untouched = param_new_constant(2, 2, 0.5f);
    Mat unsafe_values = param_values(unsafe);
    Mat untouched_values = param_values(untouched);
    Mat unsafe_gradient = param_gradient(unsafe);
    Mat untouched_gradient = param_gradient(untouched);

    unsafe_values.vals[3] = FLT_MAX / 4.0f;
    untouched_values.vals[3] = FLT_MAX / 4.0f;
    for (size_t i = 0; i < mat_size(unsafe_gradient); i++) {
        unsafe_gradient.vals[i] = 0.25f;
        untouched_gradient.vals[i] = 0.25f;
    }

    AdamW overflowing = valid;

    overflowing.learning_rate = 8.0f;
    overflowing.weight_decay = 1.0f;
    expect(param_adamw_step(unsafe, overflowing, 1) != 0,
           "AdamW rejects an update whose later element would overflow");
    expect(memcmp(unsafe_values.vals, untouched_values.vals,
                  mat_size(unsafe_values) * sizeof *unsafe_values.vals) == 0,
           "unsafe arithmetic is rejected before an earlier value moves");

    unsafe_values.vals[3] = untouched_values.vals[3] = 0.5f;
    expect(param_adamw_step(unsafe, valid, 1) == 0
           && param_adamw_step(untouched, valid, 1) == 0,
           "a repaired parameter accepts the next valid update");
    expect(memcmp(unsafe_values.vals, untouched_values.vals,
                  mat_size(unsafe_values) * sizeof *unsafe_values.vals) == 0,
           "rejection leaves optimizer history unchanged after repair");

    param_free(untouched);
    param_free(unsafe);
    param_free(control);
    param_free(tested);
}

int main(void)
{
    check_adamw();
    check_gradient_tools();
    check_parameter_io();
    check_rejection_contracts();

    if (failures != 0) {
        fprintf(stderr, "check-08: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-08: all %d optimizer checks passed\n", checks);
    return EXIT_SUCCESS;
}
