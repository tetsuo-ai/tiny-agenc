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

typedef struct {
    AdamW  opt;
    size_t count;
    float  correction1;
    float  correction2;
    float  decay;
    double safe;
    double inverse_correction1;
    double inverse_correction2;
    double inverse_epsilon;
} PreparedAdamW;

static int adamw_corrections_valid(const PreparedAdamW *step)
{
    return finite_float(step->correction1) && step->correction1 > 0.0f
        && finite_float(step->correction2) && step->correction2 > 0.0f;
}

static int adamw_prepare(PreparedAdamW *prepared, const Param *p,
                         AdamW opt, int step)
{
    if (!param_adamw_recipe_valid(opt, step))
        return 0;

    prepared->opt = opt;
    prepared->count = mat_size(p->values);
    prepared->decay = is_matrix(p) ? opt.weight_decay : 0.0f;
    prepared->correction1 = 1.0f - powf(opt.beta1, (float)step);
    prepared->correction2 = 1.0f - powf(opt.beta2, (float)step);

    if (!adamw_corrections_valid(prepared))
        return 0;
    prepared->safe = (double)FLT_MAX / 4.0;
    prepared->inverse_correction1 = 1.0 / (double)prepared->correction1;
    prepared->inverse_correction2 = 1.0 / (double)prepared->correction2;
    prepared->inverse_epsilon = 1.0 / (double)prepared->opt.epsilon;
    return 1;
}

static int adamw_stored_entry_valid(const Param *p, size_t i,
                                    float gradient)
{
    return finite_float(gradient) && finite_float(p->values.vals[i])
        && finite_float(p->first_moment[i])
        && finite_float(p->second_moment[i])
        && p->second_moment[i] >= 0.0f;
}

static int adamw_candidate_moments_valid(const PreparedAdamW *step,
                                         float first, float second)
{
    return finite_float(first) && finite_float(second) && second >= 0.0f
        && fabs((double)first) <= step->safe
        && (double)second <= step->safe;
}

static int adamw_update_bounds_valid(const Param *p,
                                     const PreparedAdamW *step, size_t i,
                                     float first, float second)
{
    double variance_bound = (double)second * step->inverse_correction2;
    double smoothed_bound =
        fabs((double)first) * step->inverse_correction1;
    double ratio_bound = smoothed_bound * step->inverse_epsilon;
    double decay_bound =
        (double)step->decay * fabs((double)p->values.vals[i]);
    double direction_bound = ratio_bound + decay_bound;
    double change_bound =
        (double)step->opt.learning_rate * direction_bound;

    return (double)step->opt.epsilon <= step->safe / 2.0
        && variance_bound <= step->safe
        && smoothed_bound <= step->safe
        && ratio_bound <= step->safe
        && decay_bound <= step->safe
        && direction_bound <= step->safe
        && change_bound <= step->safe
        && fabs((double)p->values.vals[i]) + change_bound <= step->safe;
}

static int adamw_inputs_valid(const Param *p, const PreparedAdamW *step)
{
    for (size_t i = 0; i < step->count; i++) {
        float gradient = p->gradient[i];
        float first = step->opt.beta1 * p->first_moment[i]
                    + (1.0f - step->opt.beta1) * gradient;
        float second = step->opt.beta2 * p->second_moment[i]
                     + (1.0f - step->opt.beta2) * gradient * gradient;

        if (!adamw_stored_entry_valid(p, i, gradient)
            || !adamw_candidate_moments_valid(step, first, second)
            || !adamw_update_bounds_valid(p, step, i, first, second))
            return 0;
    }
    return 1;
}

static void adamw_apply_entry(Param *p, const PreparedAdamW *step, size_t i)
{
    float gradient = p->gradient[i];

    p->first_moment[i]  = step->opt.beta1 * p->first_moment[i]
                        + (1.0f - step->opt.beta1) * gradient;
    p->second_moment[i] = step->opt.beta2 * p->second_moment[i]
                        + (1.0f - step->opt.beta2) * gradient * gradient;

    float smoothed = p->first_moment[i] / step->correction1;
    float spread   = sqrtf(p->second_moment[i] / step->correction2);

    p->values.vals[i] -= step->opt.learning_rate
                       * (smoothed / (spread + step->opt.epsilon)
                          + step->decay * p->values.vals[i]);
}

static void adamw_apply(Param *p, const PreparedAdamW *step)
{
    for (size_t i = 0; i < step->count; i++)
        adamw_apply_entry(p, step, i);
}

int param_adamw_step(Param *p, AdamW opt, int step)
{
    PreparedAdamW prepared;

    if (!adamw_prepare(&prepared, p, opt, step)
        || !adamw_inputs_valid(p, &prepared))
        return -1;
    adamw_apply(p, &prepared);
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
