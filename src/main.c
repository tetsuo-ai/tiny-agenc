/*
 * main.c -- the command line.
 *
 *   tiny-agenc train  --data corpus.txt [options]
 *   tiny-agenc sample --model tiny-agenc.bin [--prompt TEXT] [options]
 *
 * `train` fits a model to a text file and narrates its progress: a
 * loss line every LOSS_INTERVAL steps and a fresh sample of the
 * model's writing every SAMPLE_INTERVAL steps, so you can watch noise
 * become language.  `sample` reloads a checkpoint and continues a
 * prompt; the text goes to stdout, the banner to stderr, so output
 * pipes cleanly.  This is the only file that talks to a terminal and
 * the only file allowed to die on bad input.
 */
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dataset.h"
#include "evaluation.h"
#include "model.h"
#include "rng.h"
#include "tokenizer.h"
#include "util.h"
#include "version.h"

enum {
    DEFAULT_LAYER_COUNT = 4,
    DEFAULT_HEAD_COUNT  = 4,
    DEFAULT_D_MODEL     = 128,
    DEFAULT_BLOCK_SIZE  = TINY_AGENC_DEFAULT_BLOCK_SIZE,
    DEFAULT_BATCH_SIZE  = TINY_AGENC_DEFAULT_BATCH_SIZE,
    DEFAULT_STEP_COUNT  = 5000,
    MAX_STEP_COUNT      = 1 << 30,
    LOSS_INTERVAL       = 50,      /* steps between loss reports */
    SAMPLE_INTERVAL     = 250,     /* steps between generated samples */
    SAMPLE_LENGTH       = 200,     /* characters per training-time sample */
    CHECKPOINT_INTERVAL = 1000,    /* steps between saves */
    VALIDATION_BATCHES  = TINY_AGENC_VALIDATION_BATCHES,
    DEFAULT_TAIL_LENGTH = 400,     /* characters `sample` generates */
    MAX_TAIL_LENGTH     = 1 << 24,
};

static const float              DEFAULT_LEARNING_RATE   = 1e-3f;
static const float              DEFAULT_TEMPERATURE     = 0.8f;
static const unsigned long long DEFAULT_SEED = TINY_AGENC_DEFAULT_SEED;
/* Keep progress-sample draws from changing which corpus windows train next. */
static const unsigned long long SAMPLE_SEED_OFFSET =
    TINY_AGENC_SAMPLE_SEED_OFFSET;
/* Held-out windows are fixed independently of both training and sampling. */
static const unsigned long long VALIDATION_SEED_OFFSET =
    TINY_AGENC_VALIDATION_SEED_OFFSET;
static const double             MILLISECONDS_PER_SECOND = 1e3;

/* Keep the remaining AdamW constants fixed so the teaching CLI stays
 * focused; Chapter 8 records the policy. */
static const float ADAM_BETA1   = 0.9f;
static const float ADAM_BETA2   = 0.999f;
static const float ADAM_EPSILON = 1e-8f;
static const float WEIGHT_DECAY = 0.01f;

static void print_usage(FILE *stream)
{
    fputs("tiny-agenc: usage:\n"
          "  tiny-agenc train --data FILE [--out FILE] [--steps N]\n"
          "                   [--val-data FILE]\n"
          "                   [--layers N] [--heads N] [--width N] [--block N]\n"
          "                   [--batch N] [--lr F] [--seed N]\n"
          "  tiny-agenc sample --model FILE [--prompt TEXT] [--length N]\n"
          "                   [--temperature F] [--seed N]\n"
          "  tiny-agenc --help\n"
          "  tiny-agenc --version\n", stream);
}

_Noreturn static void usage_error(const char *format, ...)
{
    if (format != NULL) {
        va_list arguments;

        va_start(arguments, format);
        fputs("tiny-agenc: ", stderr);
        vfprintf(stderr, format, arguments);
        fputc('\n', stderr);
        va_end(arguments);
    }
    print_usage(stderr);
    exit(EXIT_FAILURE);
}

_Noreturn static void invalid_option(const char *command, int argc, char **argv)
{
    const char *option =
        optind > 0 && optind <= argc ? argv[optind - 1] : "(unknown)";

    usage_error("%s: invalid option '%s'", command, option);
}

/* -------- small parsers that refuse garbage -------- */

static int parse_int(const char *text, int min, int max, const char *what)
{
    char *end;
    long  value = strtol(text, &end, 10);

    if (*text == '\0' || *end != '\0' || value < min || value > max)
        die("%s wants an integer in [%d, %d], not '%s'", what, min, max, text);
    return (int)value;
}

/* Flags are human-scale numbers.  Inf and NaN are refused by bit
 * pattern -- exponent all ones -- the one float test -ffast-math can
 * never fold away; the range check then handles finite garbage.
 * (Under -ffast-math's flush-to-zero, positive subnormals compare
 * equal to zero and are rejected too: intentional, since they would
 * train as zero anyway.) */
static const float    MAX_FLAG_VALUE      = 1e6f;
static const uint32_t FLOAT_EXPONENT_BITS = 0x7F800000u;

static int is_finite_number(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & FLOAT_EXPONENT_BITS) != FLOAT_EXPONENT_BITS;
}

static float parse_positive(const char *text, const char *what)
{
    char *end;
    float value = strtof(text, &end);

    if (*text == '\0' || *end != '\0' || !is_finite_number(value)
        || value <= 0.0f || value > MAX_FLAG_VALUE)
        die("%s wants a positive number at most %g, not '%s'",
            what, (double)MAX_FLAG_VALUE, text);
    return value;
}

static unsigned long long parse_seed(const char *text)
{
    char *end;

    /* Digits only: strtoull would happily negate '-5' into a huge
     * value and silently clamp overflow, both lies about the seed. */
    errno = 0;

    unsigned long long value = strtoull(text, &end, 10);

    if (!isdigit((unsigned char)text[0]) || *end != '\0' || errno == ERANGE)
        die("--seed wants a whole number, not '%s'", text);
    return value;
}

/* -------- shared helpers -------- */

static AdamW adamw_with_rate(float learning_rate)
{
    AdamW opt = { learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, WEIGHT_DECAY };

    return opt;
}

static int paths_name_same_file(const char *first, const char *second)
{
    if (strcmp(first, second) == 0)
        return 1;

    struct stat first_status;
    struct stat second_status;

    if (stat(first, &first_status) != 0 || stat(second, &second_status) != 0)
        return 0;
    return first_status.st_dev == second_status.st_dev
        && first_status.st_ino == second_status.st_ino;
}

static void print_text(const Tokenizer *tk, const int *ids, int count)
{
    for (int i = 0; i < count; i++)
        putchar(tokenizer_decode(tk, ids[i]));
    putchar('\n');
}

static void print_architecture(FILE *stream, const Model *m)
{
    ModelConfig cfg = model_config(m);
    ModelMemory memory;

    if (!model_memory_requirements(cfg, &memory))
        die("cannot calculate model memory requirements");
    fprintf(stream,
            "tiny-agenc: %zu parameters | %.1f MiB buffers | "
            "vocab %d | %d layers x %d heads x %d wide\n",
            model_parameter_count(m),
            (double)memory.total_bytes / (1024.0 * 1024.0),
            cfg.vocab_size, cfg.layer_count, cfg.head_count, cfg.d_model);
}

/* Seed generation with a newline: the natural "start of a line" state
 * for a model trained on line-structured text. */
static int newline_id(const Tokenizer *tk)
{
    int id;

    if (tokenizer_encode(tk, &id, "\n", 1) != 1)
        die("corpus vocabulary has no newline to seed sampling with");
    return id;
}

/* -------- train -------- */

typedef struct {
    const char *data_path;
    const char *validation_path;
    const char *out_path;
    int         steps;
    ModelConfig cfg;   /* vocab_size filled in after reading the corpus */
    float       learning_rate;
    unsigned long long seed;
} TrainOptions;

typedef struct {
    Tokenizer *tokenizer;
    Dataset   *training_data;
    Dataset   *validation_data;
    Model     *model;
    Rng       *batch_rng;
    Rng       *sample_rng;
} TrainingResources;

static void reject_training_path_collisions(const TrainOptions *options)
{
    if (paths_name_same_file(options->data_path, options->out_path))
        die("--out must not name the --data file");
    if (options->validation_path != NULL
        && paths_name_same_file(options->validation_path, options->out_path))
        die("--out must not name the --val-data file");
    if (options->validation_path != NULL
        && paths_name_same_file(options->data_path,
                                options->validation_path))
        die("--data and --val-data must name distinct files");
}

static const struct option TRAIN_FLAGS[] = {
    { "data",   required_argument, NULL, 'd' },
    { "val-data", required_argument, NULL, 'v' },
    { "out",    required_argument, NULL, 'o' },
    { "steps",  required_argument, NULL, 's' },
    { "layers", required_argument, NULL, 'l' },
    { "heads",  required_argument, NULL, 'h' },
    { "width",  required_argument, NULL, 'w' },
    { "block",  required_argument, NULL, 'k' },
    { "batch",  required_argument, NULL, 'b' },
    { "lr",     required_argument, NULL, 'r' },
    { "seed",   required_argument, NULL, 'x' },
    { NULL, 0, NULL, 0 },
};

static char *read_training_text(const TrainOptions *options, size_t *length)
{
    char *text;
    FileSlurpStatus status =
        file_slurp_bounded(options->data_path, dataset_max_text_bytes(),
                           &text, length);

    if (status == FILE_SLURP_TOO_LARGE)
        die("corpus %s exceeds the platform limit of %zu bytes",
            options->data_path, dataset_max_text_bytes());
    if (status != FILE_SLURP_OK)
        die("cannot read corpus %s (try `make corpus` or `make data`)",
            options->data_path);
    return text;
}

static Dataset *load_training_data(const TrainOptions *options,
                                   Tokenizer **tokenizer)
{
    size_t length;
    char  *text = read_training_text(options, &length);

    *tokenizer = tokenizer_new(text, length);
    (void)newline_id(*tokenizer);

    Dataset *dataset = dataset_new(*tokenizer, text, length);

    free(text);
    if (dataset == NULL)
        die("corpus %s cannot fit in a token buffer on this platform",
            options->data_path);
    if (dataset_token_count(dataset) < (size_t)options->cfg.block_size + 1)
        die("corpus %s is smaller than one training window",
            options->data_path);
    return dataset;
}

static char *read_validation_text(const TrainOptions *options, size_t *length)
{
    char *text;
    FileSlurpStatus status =
        file_slurp_bounded(options->validation_path,
                           dataset_max_text_bytes(), &text, length);

    if (status == FILE_SLURP_TOO_LARGE)
        die("validation corpus %s exceeds the platform limit of %zu bytes",
            options->validation_path, dataset_max_text_bytes());
    if (status != FILE_SLURP_OK)
        die("cannot read validation corpus %s", options->validation_path);
    return text;
}

static Dataset *load_validation_data(const TrainOptions *options,
                                     const Tokenizer *tokenizer)
{
    if (options->validation_path == NULL)
        return NULL;

    size_t length;
    char  *text    = read_validation_text(options, &length);
    Dataset *dataset = dataset_new(tokenizer, text, length);

    free(text);
    if (dataset == NULL)
        die("validation corpus %s cannot fit in a token buffer on this platform",
            options->validation_path);
    if (dataset_token_count(dataset) != length)
        die("validation corpus %s contains bytes absent from training data",
            options->validation_path);
    if (dataset_token_count(dataset) < (size_t)options->cfg.block_size + 1)
        die("validation corpus %s is smaller than one training window",
            options->validation_path);
    return dataset;
}

static void validate_training_model(TrainOptions *options,
                                    const Tokenizer *tokenizer)
{
    options->cfg.vocab_size = tokenizer_vocab_size(tokenizer);
    if (!model_config_valid(options->cfg))
        die("impossible model configuration: --width must be divisible by --heads "
            "and --batch x --block must stay within %d tokens",
            MODEL_MAX_TOKENS_PER_PASS);

    ModelMemory memory;

    if (!model_memory_requirements(options->cfg, &memory))
        die("model memory requirements overflow this platform");
    if (memory.total_bytes > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES)
        die("model needs %.1f MiB of buffers; the CLI limit is %.0f MiB",
            (double)memory.total_bytes / (1024.0 * 1024.0),
            (double)MODEL_MAX_CHECKPOINT_RESIDENT_BYTES / (1024.0 * 1024.0));
}

static TrainingResources prepare_training_resources(TrainOptions *options)
{
    TrainingResources resources = { 0 };

    resources.training_data =
        load_training_data(options, &resources.tokenizer);
    resources.validation_data =
        load_validation_data(options, resources.tokenizer);
    validate_training_model(options, resources.tokenizer);
    resources.model      = model_new(options->cfg, options->seed);
    resources.batch_rng  = rng_new(options->seed);
    resources.sample_rng = rng_new(options->seed + SAMPLE_SEED_OFFSET);
    return resources;
}

static void print_training_summary(const TrainingResources *resources,
                                   const TrainOptions *options)
{
    print_architecture(stdout, resources->model);
    printf("tiny-agenc: %zu tokens of training data from %s\n",
           dataset_token_count(resources->training_data), options->data_path);
    if (resources->validation_data != NULL)
        printf("tiny-agenc: %zu tokens of validation data from %s\n",
               dataset_token_count(resources->validation_data),
               options->validation_path);
}

static void free_training_resources(TrainingResources resources)
{
    rng_free(resources.sample_rng);
    rng_free(resources.batch_rng);
    model_free(resources.model);
    if (resources.validation_data != NULL)
        dataset_free(resources.validation_data);
    dataset_free(resources.training_data);
    tokenizer_free(resources.tokenizer);
}

typedef struct {
    int *inputs;
    int *targets;
    int  count;
} ValidationBatches;

static ValidationBatches prepare_validation(const Dataset *ds,
                                            const TrainOptions *options)
{
    ValidationBatches validation = { 0 };

    if (ds == NULL)
        return validation;

    int    span  = options->cfg.batch_size * options->cfg.block_size;
    size_t count = (size_t)VALIDATION_BATCHES * (size_t)span;
    Rng   *rng   = rng_new(options->seed + VALIDATION_SEED_OFFSET);

    validation.inputs  = emalloc(count * sizeof *validation.inputs);
    validation.targets = emalloc(count * sizeof *validation.targets);
    validation.count   = VALIDATION_BATCHES;

    for (int batch = 0; batch < validation.count; batch++)
        dataset_batch(ds, rng,
                      validation.inputs + (size_t)batch * (size_t)span,
                      validation.targets + (size_t)batch * (size_t)span,
                      options->cfg.batch_size, options->cfg.block_size);
    rng_free(rng);
    return validation;
}

static float measure_validation(Model *m, ValidationBatches validation,
                                const TrainOptions *options)
{
    int    span  = options->cfg.batch_size * options->cfg.block_size;
    double total = 0.0;

    for (int batch = 0; batch < validation.count; batch++)
        total += model_forward(m,
                               validation.inputs + (size_t)batch * (size_t)span,
                               validation.targets + (size_t)batch * (size_t)span,
                               options->cfg.batch_size, options->cfg.block_size);
    return (float)(total / validation.count);
}

static void free_validation(ValidationBatches validation)
{
    free(validation.inputs);
    free(validation.targets);
}

static void print_training_sample(Model *m, const Tokenizer *tk, Rng *rng, int step)
{
    int ids[1 + SAMPLE_LENGTH];

    ids[0] = newline_id(tk);
    model_sample(m, rng, ids, 1, 1 + SAMPLE_LENGTH, DEFAULT_TEMPERATURE);
    printf("---- sample at step %d ----\n", step);
    print_text(tk, ids + 1, SAMPLE_LENGTH);
    printf("---------------------------\n");
}

typedef struct {
    int   *inputs;
    int   *targets;
    AdamW  optimizer;
} TrainingStepState;

typedef struct {
    ValidationBatches validation;
    double            clock;
    int               timed_steps;
} TrainingObservationState;

static TrainingStepState prepare_training_step_state(
    const TrainOptions *options)
{
    int span = options->cfg.batch_size * options->cfg.block_size;
    TrainingStepState state;

    state.inputs     = emalloc((size_t)span * sizeof *state.inputs);
    state.targets    = emalloc((size_t)span * sizeof *state.targets);
    state.optimizer  = adamw_with_rate(options->learning_rate);
    return state;
}

static void free_training_step_state(TrainingStepState state)
{
    free(state.inputs);
    free(state.targets);
}

static TrainingObservationState prepare_training_observation(
    const TrainingResources *resources, const TrainOptions *options)
{
    TrainingObservationState state;

    state.validation =
        prepare_validation(resources->validation_data, options);
    state.clock       = time_seconds();
    state.timed_steps = 0;
    return state;
}

static void free_training_observation(TrainingObservationState state)
{
    free_validation(state.validation);
}

static float run_training_step(TrainingResources *resources,
                               TrainingStepState *state,
                               const TrainOptions *options, int step)
{
    dataset_batch(resources->training_data, resources->batch_rng,
                  state->inputs, state->targets,
                  options->cfg.batch_size, options->cfg.block_size);
    model_zero_gradients(resources->model);

    float loss =
        model_forward(resources->model, state->inputs, state->targets,
                      options->cfg.batch_size, options->cfg.block_size);

    model_backward(resources->model);
    if (model_step(resources->model, state->optimizer, step) != 0)
        die("non-finite gradient or invalid optimizer update at step %d",
            step);
    return loss;
}

static void restart_training_timer(TrainingObservationState *state)
{
    state->clock       = time_seconds();
    state->timed_steps = 0;
}

static void print_training_report(TrainingResources *resources,
                                  TrainingObservationState *state,
                                  const TrainOptions *options,
                                  int step, float loss, double milliseconds)
{
    if (state->validation.count > 0) {
        float validation_loss =
            measure_validation(resources->model, state->validation, options);

        printf("step %5d/%d | loss %.4f | val %.4f | %6.1f ms/step\n",
               step, options->steps, (double)loss,
               (double)validation_loss, milliseconds);
    } else {
        printf("step %5d/%d | loss %.4f | %6.1f ms/step\n",
               step, options->steps, (double)loss, milliseconds);
    }
}

static void report_training_if_due(TrainingResources *resources,
                                   TrainingObservationState *state,
                                   const TrainOptions *options,
                                   int step, float loss)
{
    if (step == 1 || step % LOSS_INTERVAL == 0) {
        double now = time_seconds();
        double milliseconds =
            MILLISECONDS_PER_SECOND
            * (now - state->clock) / state->timed_steps;

        print_training_report(resources, state, options, step, loss,
                              milliseconds);
        restart_training_timer(state);
    }
}

static void sample_training_if_due(TrainingResources *resources,
                                   TrainingObservationState *state, int step)
{
    if (step % SAMPLE_INTERVAL == 0) {
        print_training_sample(resources->model, resources->tokenizer,
                              resources->sample_rng, step);
        restart_training_timer(state);
    }
}

static void save_training_if_due(TrainingResources *resources,
                                 TrainingObservationState *state,
                                 const TrainOptions *options, int step)
{
    if (step % CHECKPOINT_INTERVAL == 0 || step == options->steps) {
        ModelSaveResult saved =
            model_save_durable(resources->model, resources->tokenizer,
                               options->out_path);

        if (saved == MODEL_SAVE_NOT_COMMITTED)
            die("cannot write checkpoint %s", options->out_path);
        if (saved == MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED)
            die("checkpoint %s was committed, but directory finalization "
                "failed; durability is unconfirmed", options->out_path);
        restart_training_timer(state);
    }
}

static void train_loop(TrainingResources *resources,
                       const TrainOptions *options)
{
    TrainingStepState step_state = prepare_training_step_state(options);
    TrainingObservationState observation =
        prepare_training_observation(resources, options);

    for (int step = 1; step <= options->steps; step++) {
        float loss =
            run_training_step(resources, &step_state, options, step);

        observation.timed_steps++;
        report_training_if_due(resources, &observation, options, step, loss);
        sample_training_if_due(resources, &observation, step);
        save_training_if_due(resources, &observation, options, step);
    }
    free_training_observation(observation);
    free_training_step_state(step_state);
}

static int run_train(TrainOptions options)
{
    reject_training_path_collisions(&options);

    TrainingResources resources = prepare_training_resources(&options);

    print_training_summary(&resources, &options);
    train_loop(&resources, &options);
    printf("tiny-agenc: checkpoint saved to %s\n", options.out_path);
    free_training_resources(resources);
    return EXIT_SUCCESS;
}

static int parse_train(int argc, char **argv)
{
    TrainOptions options = {
        .data_path     = NULL,
        .validation_path = NULL,
        .out_path      = "trained.bin",
        .steps         = DEFAULT_STEP_COUNT,
        .cfg           = {
            .vocab_size  = 0,   /* set from the corpus */
            .block_size  = DEFAULT_BLOCK_SIZE,
            .d_model     = DEFAULT_D_MODEL,
            .head_count  = DEFAULT_HEAD_COUNT,
            .layer_count = DEFAULT_LAYER_COUNT,
            .batch_size  = DEFAULT_BATCH_SIZE,
        },
        .learning_rate = DEFAULT_LEARNING_RATE,
        .seed          = DEFAULT_SEED,
    };
    int letter;

    opterr = 0;
    while ((letter = getopt_long(argc, argv, "", TRAIN_FLAGS, NULL)) != -1) {
        switch (letter) {
        case 'd': options.data_path = optarg;                                                          break;
        case 'v': options.validation_path = optarg;                                                    break;
        case 'o': options.out_path  = optarg;                                                          break;
        case 's': options.steps = parse_int(optarg, 1, MAX_STEP_COUNT, "--steps");                     break;
        case 'l': options.cfg.layer_count = parse_int(optarg, 1, MODEL_MAX_LAYER_COUNT, "--layers");   break;
        case 'h': options.cfg.head_count = parse_int(optarg, 1, MODEL_MAX_HEAD_COUNT, "--heads");      break;
        case 'w': options.cfg.d_model = parse_int(optarg, 1, MODEL_MAX_D_MODEL, "--width");            break;
        case 'k': options.cfg.block_size = parse_int(optarg, 1, MODEL_MAX_BLOCK_SIZE, "--block");      break;
        case 'b': options.cfg.batch_size = parse_int(optarg, 1, MODEL_MAX_TOKENS_PER_PASS, "--batch"); break;
        case 'r': options.learning_rate = parse_positive(optarg, "--lr");                              break;
        case 'x': options.seed = parse_seed(optarg);                                                   break;
        default:  invalid_option("train", argc, argv);
        }
    }
    if (options.data_path == NULL)
        usage_error("train requires --data FILE");
    if (optind != argc)
        usage_error("train: unexpected argument '%s'", argv[optind]);
    return run_train(options);
}

/* -------- sample -------- */

typedef struct {
    const char *model_path;
    const char *prompt;
    int         length;
    float       temperature;
    unsigned long long seed;
} SampleOptions;

static const struct option SAMPLE_FLAGS[] = {
    { "model",       required_argument, NULL, 'm' },
    { "prompt",      required_argument, NULL, 'p' },
    { "length",      required_argument, NULL, 'n' },
    { "temperature", required_argument, NULL, 't' },
    { "seed",        required_argument, NULL, 'x' },
    { NULL, 0, NULL, 0 },
};

static int run_sample(SampleOptions options)
{
    Tokenizer *tk;
    Model     *m = model_load(&tk, options.model_path);

    if (m == NULL)
        die("cannot load checkpoint %s (train one first?)", options.model_path);
    print_architecture(stderr, m);

    size_t prompt_length = strlen(options.prompt);
    int   *ids           = emalloc((1 + prompt_length + (size_t)options.length) * sizeof *ids);

    /* A newline before the prompt puts the model at start-of-line, the
     * state every line of its corpus began from. */
    ids[0] = newline_id(tk);

    int  known = 1 + (int)tokenizer_encode(tk, ids + 1, options.prompt, prompt_length);
    int  total = known + options.length;
    Rng *rng   = rng_new(options.seed);

    model_sample(m, rng, ids, known, total, options.temperature);
    print_text(tk, ids + 1, total - 1);   /* everything but the seed newline */

    rng_free(rng);
    free(ids);
    model_free(m);
    tokenizer_free(tk);
    return EXIT_SUCCESS;
}

static int parse_sample(int argc, char **argv)
{
    SampleOptions options = {
        .model_path  = NULL,
        .prompt      = "",
        .length      = DEFAULT_TAIL_LENGTH,
        .temperature = DEFAULT_TEMPERATURE,
        .seed        = DEFAULT_SEED,
    };
    int letter;

    opterr = 0;
    while ((letter = getopt_long(argc, argv, "", SAMPLE_FLAGS, NULL)) != -1) {
        switch (letter) {
        case 'm': options.model_path = optarg;                                        break;
        case 'p': options.prompt     = optarg;                                        break;
        case 'n': options.length = parse_int(optarg, 1, MAX_TAIL_LENGTH, "--length"); break;
        case 't': options.temperature = parse_positive(optarg, "--temperature");      break;
        case 'x': options.seed = parse_seed(optarg);                                  break;
        default:  invalid_option("sample", argc, argv);
        }
    }
    if (optind != argc)
        usage_error("sample: unexpected argument '%s'", argv[optind]);
    if (options.model_path == NULL)
        usage_error("sample requires --model FILE");
    return run_sample(options);
}

/* On the measured SMT machine, one thread per physical core performed
 * best.  Unless the user overrides it, half the reported logical
 * processor count is a practical default that also limits fork/join
 * overhead on small calls. */
static void choose_thread_count(void)
{
#ifdef _OPENMP
    int half = omp_get_num_procs() / 2;

    if (getenv("OMP_NUM_THREADS") == NULL && half > 0)
        omp_set_num_threads(half);
#endif
}

int main(int argc, char **argv)
{
    /* Training narrates as it goes; line-buffering keeps the story
     * streaming even when stdout is a file being tailed. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    choose_thread_count();
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        print_usage(stdout);
        return EXIT_SUCCESS;
    }
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        printf("tiny-agenc %s\n", TINY_AGENC_VERSION);
        return EXIT_SUCCESS;
    }
    if (argc < 2)
        usage_error(NULL);
    if (strcmp(argv[1], "train") == 0)
        return parse_train(argc - 1, argv + 1);
    if (strcmp(argv[1], "sample") == 0)
        return parse_sample(argc - 1, argv + 1);
    usage_error("unknown command '%s'", argv[1]);
}
