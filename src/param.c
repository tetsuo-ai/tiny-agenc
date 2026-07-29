#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "param.h"
#include "util.h"

struct Param {
    Mat    values;
    float *gradient;
    float *first_moment;    /* running average of gradients */
    float *second_moment;   /* running average of squared gradients */
};

enum { PARAM_BUFFERS = 4 };   /* values, gradient, and two moments */

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & 0x7F800000u) != 0x7F800000u;
}

/* One backing allocation carved four ways; the gradient region starts zeroed. */
static Param *param_new(int rows, int cols)
{
    if (rows < 1 || cols < 1
        || (size_t)rows > SIZE_MAX / (size_t)cols)
        return NULL;

    size_t count = (size_t)rows * (size_t)cols;

    if (count > SIZE_MAX / PARAM_BUFFERS
        || PARAM_BUFFERS * count > SIZE_MAX / sizeof(float))
        return NULL;

    Param  *p     = emalloc(sizeof *p);
    float  *store = ecalloc(PARAM_BUFFERS * count, sizeof *store);

    p->values        = mat_make(store, rows, cols);
    p->gradient      = store + count;
    p->first_moment  = store + 2 * count;
    p->second_moment = store + 3 * count;
    return p;
}

Param *param_new_gaussian(int rows, int cols, float stddev, Rng *rng)
{
    Param  *p     = param_new(rows, cols);

    if (p == NULL)
        return NULL;
    size_t count = mat_size(p->values);
    for (size_t i = 0; i < count; i++)
        p->values.vals[i] = stddev * rng_gaussian(rng);
    return p;
}

Param *param_new_constant(int rows, int cols, float value)
{
    Param  *p     = param_new(rows, cols);

    if (p == NULL)
        return NULL;
    size_t count = mat_size(p->values);
    for (size_t i = 0; i < count; i++)
        p->values.vals[i] = value;
    return p;
}

Mat param_values(const Param *p)
{
    return p->values;
}

Mat param_gradient(const Param *p)
{
    return mat_make(p->gradient, p->values.rows, p->values.cols);
}

void param_zero_gradient(Param *p)
{
    memset(p->gradient, 0, mat_size(p->values) * sizeof *p->gradient);
}

int param_gradient_is_finite(const Param *p)
{
    size_t count = mat_size(p->values);

    for (size_t i = 0; i < count; i++)
        if (!finite_float(p->gradient[i]))
            return 0;
    return 1;
}

float param_gradient_norm_squared(const Param *p)
{
    size_t count = mat_size(p->values);
    double total = 0.0;

    for (size_t i = 0; i < count; i++)
        total += (double)p->gradient[i] * (double)p->gradient[i];
    return (float)total;
}

void param_scale_gradient(Param *p, float factor)
{
    size_t count = mat_size(p->values);

    for (size_t i = 0; i < count; i++)
        p->gradient[i] *= factor;
}

/* Shapes with more than one row and column get weight decay; gains,
 * biases, and other vectors do not. */
static int is_matrix(const Param *p)
{
    return p->values.rows > 1 && p->values.cols > 1;
}

int param_adamw_recipe_valid(AdamW opt, int step)
{
    return step >= 1
        && finite_float(opt.learning_rate) && opt.learning_rate >= 0.0f
        && finite_float(opt.beta1) && opt.beta1 >= 0.0f && opt.beta1 < 1.0f
        && finite_float(opt.beta2) && opt.beta2 >= 0.0f && opt.beta2 < 1.0f
        && finite_float(opt.epsilon) && opt.epsilon > 0.0f
        && finite_float(opt.weight_decay) && opt.weight_decay >= 0.0f;
}

static int adamw_inputs_valid(const Param *p, AdamW opt,
                              float correction1, float correction2,
                              float decay)
{
    size_t count = mat_size(p->values);
    const double safe = (double)FLT_MAX / 4.0;
    const double inverse_correction1 = 1.0 / (double)correction1;
    const double inverse_correction2 = 1.0 / (double)correction2;
    const double inverse_epsilon = 1.0 / (double)opt.epsilon;

    for (size_t i = 0; i < count; i++) {
        float gradient = p->gradient[i];
        float first = opt.beta1 * p->first_moment[i]
                    + (1.0f - opt.beta1) * gradient;
        float second = opt.beta2 * p->second_moment[i]
                     + (1.0f - opt.beta2) * gradient * gradient;

        if (!finite_float(gradient) || !finite_float(p->values.vals[i])
            || !finite_float(p->first_moment[i])
            || !finite_float(p->second_moment[i])
            || p->second_moment[i] < 0.0f
            || !finite_float(first) || !finite_float(second)
            || second < 0.0f
            || fabs((double)first) > safe
            || (double)second > safe)
            return 0;

        /* Conservative magnitude bounds prove that every operation in
         * the original float update below stays finite.  This avoids a
         * second square root per element while preserving its exact
         * normal-path arithmetic. */
        double variance_bound = (double)second * inverse_correction2;
        double smoothed_bound =
            fabs((double)first) * inverse_correction1;
        double ratio_bound = smoothed_bound * inverse_epsilon;
        double decay_bound =
            (double)decay * fabs((double)p->values.vals[i]);
        double direction_bound = ratio_bound + decay_bound;
        double change_bound =
            (double)opt.learning_rate * direction_bound;

        if ((double)opt.epsilon > safe / 2.0
            || variance_bound > safe
            || smoothed_bound > safe
            || ratio_bound > safe
            || decay_bound > safe
            || direction_bound > safe
            || change_bound > safe
            || fabs((double)p->values.vals[i]) + change_bound > safe)
            return 0;
    }
    return 1;
}

int param_adamw_step(Param *p, AdamW opt, int step)
{
    if (!param_adamw_recipe_valid(opt, step))
        return -1;

    size_t count = mat_size(p->values);
    float  decay = is_matrix(p) ? opt.weight_decay : 0.0f;

    /* The moment averages start at zero, so early on they underestimate
     * by exactly 1 - beta^step; dividing that out unbiases them. */
    float correction1 = 1.0f - powf(opt.beta1, (float)step);
    float correction2 = 1.0f - powf(opt.beta2, (float)step);

    if (!finite_float(correction1) || correction1 <= 0.0f
        || !finite_float(correction2) || correction2 <= 0.0f)
        return -1;
    if (!adamw_inputs_valid(p, opt, correction1, correction2, decay))
        return -1;

    for (size_t i = 0; i < count; i++) {
        float gradient = p->gradient[i];

        p->first_moment[i]  = opt.beta1 * p->first_moment[i]
                            + (1.0f - opt.beta1) * gradient;
        p->second_moment[i] = opt.beta2 * p->second_moment[i]
                            + (1.0f - opt.beta2) * gradient * gradient;

        float smoothed = p->first_moment[i] / correction1;
        float spread   = sqrtf(p->second_moment[i] / correction2);

        p->values.vals[i] -= opt.learning_rate
                           * (smoothed / (spread + opt.epsilon)
                              + decay * p->values.vals[i]);
    }
    return 0;
}

static int values_are_finite(const Param *p)
{
    size_t count = mat_size(p->values);

    for (size_t i = 0; i < count; i++) {
        if (!finite_float(p->values.vals[i]))
            return 0;
    }
    return 1;
}

int param_write(const Param *p, FILE *stream)
{
    size_t count = mat_size(p->values);

    if (!values_are_finite(p))
        return -1;
    return fwrite(p->values.vals, sizeof *p->values.vals, count, stream) == count ? 0 : -1;
}

int param_read(Param *p, FILE *stream)
{
    size_t count = mat_size(p->values);

    if (fread(p->values.vals, sizeof *p->values.vals, count, stream) != count)
        return -1;
    return values_are_finite(p) ? 0 : -1;
}

void param_free(Param *p)
{
    free(p->values.vals);   /* the one allocation behind all four buffers */
    free(p);
}
