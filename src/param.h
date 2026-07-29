/*
 * param.h -- a learnable tensor.
 *
 * A Param owns four equally-sized float buffers: the values the model
 * computes with, the gradient the backward pass accumulates, and the
 * two AdamW moment estimates that smooth the update.  Keeping the
 * optimizer state with the tensor it belongs to makes a training step
 * a plain loop over Params.
 */
#ifndef TINY_AGENC_PARAM_H
#define TINY_AGENC_PARAM_H

#include <stdio.h>

#include "mat.h"
#include "rng.h"

typedef struct Param Param;

/* The AdamW recipe, passed by value: an update is a decayed average of
 * gradients, scaled down where gradients have been large. */
typedef struct {
    float learning_rate;
    float beta1;          /* decay of the gradient average */
    float beta2;          /* decay of the squared-gradient average */
    float epsilon;        /* keeps the divide finite */
    float weight_decay;   /* pull toward zero for eligible matrix shapes */
} AdamW;

/* Constructors return NULL when the shape is non-positive or its four
 * float buffers cannot be represented by size_t. */
Param *param_new_gaussian(int rows, int cols, float stddev, Rng *rng);
Param *param_new_constant(int rows, int cols, float value);

Mat    param_values(const Param *p);
Mat    param_gradient(const Param *p);

void   param_zero_gradient(Param *p);
int    param_gradient_is_finite(const Param *p);
float  param_gradient_norm_squared(const Param *p);
void   param_scale_gradient(Param *p, float factor);

/* A recipe is valid when its finite ranges make the AdamW equations
 * meaningful.  `step` counts from one. */
int    param_adamw_recipe_valid(AdamW opt, int step);

/* The update is all-or-nothing for this Param.  It returns -1 before
 * changing values or moments when the recipe, input state, or conservative
 * arithmetic bounds cannot prove that the float update will stay finite. */
int    param_adamw_step(Param *p, AdamW opt, int step);

/* Checkpoint I/O for the values only; 0 on success. */
int    param_write(const Param *p, FILE *stream);
int    param_read(Param *p, FILE *stream);

void   param_free(Param *p);

#endif
