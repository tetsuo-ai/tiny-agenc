/*
 * Chapter 13 witness: the new TAGC checkpoint is exact, checksummed,
 * durably committed, resource-bounded, and intentionally incompatible
 * with the old magic.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "model.h"
#include "param.h"
#include "tokenizer.h"

static int checks;
static int failures;
static const uint32_t CRC32_POLYNOMIAL = 0xEDB88320u;
static const uint32_t CRC32_INITIAL = 0xFFFFFFFFu;
static const uint32_t CHECKPOINT_MAGIC = 0x43474154u;
static const uint32_t LEGACY_CHECKPOINT_MAGIC = 0x4B524754u;
enum {
    CRC32_BITS_PER_BYTE = 8,
    CRC32_BUFFER_BYTES = 4096,
    CHECKPOINT_VERSION = 1,
    CORRUPTION_PAYLOAD_OFFSET = 40,
    MINIMUM_MODEL_EXTENT = 1,
    RESOURCE_BOMB_EXTENT = 1024,
    BLOCK_WEIGHT_WIDTHS = 12,
    BLOCK_VECTOR_WIDTHS = 4,
    FINAL_VECTOR_WIDTHS = 2,
    RESOURCE_VOCABULARY_SIZE = 1,
    PREVIOUS_BYTE_OFFSET = -1,
    LOW_BIT_MASK = 1,
};

static const unsigned long long FIXTURE_MODEL_SEED = 91;

typedef struct {
    Tokenizer *tokenizer;
    Model *model;
} CheckpointFixture;

typedef struct {
    char checkpoint[sizeof "/tmp/tiny-agenc-check13-XXXXXX"];
    char resource[sizeof "/tmp/tiny-agenc-check13-resource-XXXXXX"];
    char link[sizeof "/tmp/tiny-agenc-check13-link-XXXXXX"];
} CheckpointPaths;

static void expect(int condition, const char *message);
static int models_equal(const Model *left, const Model *right);
static uint32_t crc32_update(uint32_t crc, const unsigned char *bytes,
                             size_t count);
static int rewrite_checksum(const char *path);
static size_t parameter_count(ModelConfig config);
static int write_resource_fixture(const char *path, ModelConfig config);
static CheckpointPaths checkpoint_paths(void);
static int reserve_checkpoint_paths(CheckpointPaths *paths);
static void remove_checkpoint_paths(const CheckpointPaths *paths);
static CheckpointFixture checkpoint_fixture(void);
static void checkpoint_fixture_free(CheckpointFixture fixture);
static void check_save_boundaries(CheckpointFixture fixture,
                                  const CheckpointPaths *paths);
static void check_checkpoint_round_trip(CheckpointFixture fixture,
                                        const char *path);
static int corrupt_checkpoint_payload(const char *path);
static void check_corruption_rejection(const char *path);
static void check_legacy_rejection(CheckpointFixture fixture,
                                   const char *path);
static void check_resource_rejection(CheckpointFixture fixture,
                                     const char *path);
int main(void);

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-13: %s\n", message);
    failures++;
}

static int models_equal(const Model *left, const Model *right)
{
    ModelParams a = model_params(left);
    ModelParams b = model_params(right);

    if (a.count != b.count)
        return 0;
    for (int p = 0; p < a.count; p++) {
        Mat av = param_values(a.params[p]);
        Mat bv = param_values(b.params[p]);

        if (av.rows != bv.rows || av.cols != bv.cols
            || memcmp(av.vals, bv.vals,
                      mat_size(av) * sizeof *av.vals) != 0)
            return 0;
    }
    return 1;
}

static uint32_t crc32_update(uint32_t crc, const unsigned char *bytes,
                             size_t count)
{
    for (size_t i = 0; i < count; i++) {
        crc ^= bytes[i];
        for (int bit = 0; bit < CRC32_BITS_PER_BYTE; bit++)
            crc = (crc >> 1)
                ^ (CRC32_POLYNOMIAL & (0u - (crc & 1u)));
    }
    return crc;
}

static int rewrite_checksum(const char *path)
{
    FILE *stream = fopen(path, "r+b");

    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
        if (stream != NULL)
            fclose(stream);
        return -1;
    }
    long end = ftell(stream);

    if (end < (long)sizeof(uint32_t) || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return -1;
    }

    long remaining = end - (long)sizeof(uint32_t);
    unsigned char buffer[CRC32_BUFFER_BYTES];
    uint32_t crc = CRC32_INITIAL;

    while (remaining > 0) {
        size_t wanted =
            remaining < (long)sizeof buffer ? (size_t)remaining : sizeof buffer;

        if (fread(buffer, 1, wanted, stream) != wanted) {
            fclose(stream);
            return -1;
        }
        crc = crc32_update(crc, buffer, wanted);
        remaining -= (long)wanted;
    }
    crc = ~crc;

    int failed = fseek(stream, end - (long)sizeof crc, SEEK_SET) != 0
              || fwrite(&crc, sizeof crc, 1, stream) != 1;

    if (fclose(stream) != 0)
        failed = 1;
    return failed ? -1 : 0;
}

static size_t parameter_count(ModelConfig config)
{
    size_t width = (size_t)config.d_model;

    return ((size_t)config.vocab_size + (size_t)config.block_size) * width
         + (size_t)config.layer_count
             * (BLOCK_WEIGHT_WIDTHS * width * width
                + BLOCK_VECTOR_WIDTHS * width)
         + FINAL_VECTOR_WIDTHS * width;
}

static int write_resource_fixture(const char *path, ModelConfig config)
{
    int32_t header[] = {
        (int32_t)CHECKPOINT_MAGIC,
        CHECKPOINT_VERSION,
        config.vocab_size,
        config.block_size,
        config.d_model,
        config.head_count,
        config.layer_count,
        config.batch_size,
        RESOURCE_VOCABULARY_SIZE,
    };
    FILE *stream = fopen(path, "wb");
    unsigned char token = '\n';
    float zero = 0.0f;
    uint32_t checksum = 0;

    if (stream == NULL)
        return -1;
    int failed =
        fwrite(header, sizeof header[0],
               sizeof header / sizeof header[0], stream)
            != sizeof header / sizeof header[0]
        || fwrite(&token, sizeof token, 1, stream) != 1;

    for (size_t i = 0; !failed && i < parameter_count(config); i++)
        failed = fwrite(&zero, sizeof zero, 1, stream) != 1;
    if (!failed)
        failed = fwrite(&checksum, sizeof checksum, 1, stream) != 1;
    if (fclose(stream) != 0)
        failed = 1;
    return failed || rewrite_checksum(path) != 0 ? -1 : 0;
}

static CheckpointPaths checkpoint_paths(void)
{
    CheckpointPaths paths = {
        .checkpoint = "/tmp/tiny-agenc-check13-XXXXXX",
        .resource = "/tmp/tiny-agenc-check13-resource-XXXXXX",
        .link = "/tmp/tiny-agenc-check13-link-XXXXXX",
    };

    return paths;
}

static int reserve_checkpoint_paths(CheckpointPaths *paths)
{
    char *const regular_paths[] = {
        paths->checkpoint,
        paths->resource,
    };

    for (size_t i = 0;
         i < sizeof regular_paths / sizeof regular_paths[0]; i++) {
        int descriptor = mkstemp(regular_paths[i]);

        if (descriptor < 0)
            return -1;
        if (close(descriptor) != 0)
            return -1;
    }

    int link_descriptor = mkstemp(paths->link);

    if (link_descriptor < 0)
        return -1;
    if (close(link_descriptor) != 0 || remove(paths->link) != 0)
        return -1;
    return 0;
}

static void remove_checkpoint_paths(const CheckpointPaths *paths)
{
    remove(paths->checkpoint);
    remove(paths->resource);
    remove(paths->link);
}

static CheckpointFixture checkpoint_fixture(void)
{
    static const char TEXT[] = "\nabcabcabc\n";
    ModelConfig config = {
        .vocab_size = 4,
        .block_size = 4,
        .d_model = 8,
        .head_count = 2,
        .layer_count = 1,
        .batch_size = 2,
    };
    CheckpointFixture fixture = {
        .tokenizer = tokenizer_new(TEXT, sizeof TEXT - 1),
        .model = model_new(config, FIXTURE_MODEL_SEED),
    };

    return fixture;
}

static void checkpoint_fixture_free(CheckpointFixture fixture)
{
    model_free(fixture.model);
    tokenizer_free(fixture.tokenizer);
}

static void check_save_boundaries(CheckpointFixture fixture,
                                  const CheckpointPaths *paths)
{
    expect(model_save_durable(fixture.model, fixture.tokenizer,
                              paths->checkpoint) == MODEL_SAVE_DURABLE,
           "TAGC checkpoint confirms file and directory durability");
    expect(symlink(paths->checkpoint, paths->link) == 0,
           "non-regular destination fixture creates a symbolic link");
    expect(model_save_durable(fixture.model, fixture.tokenizer,
                              paths->link) == MODEL_SAVE_NOT_COMMITTED,
           "save rejects a symbolic-link destination before commit");
    expect(model_save_durable(NULL, fixture.tokenizer,
                              paths->checkpoint)
               == MODEL_SAVE_NOT_COMMITTED,
           "invalid save input reports that nothing was committed");
}

static void check_checkpoint_round_trip(CheckpointFixture fixture,
                                        const char *path)
{
    Tokenizer *loaded_tokenizer = NULL;
    Model *loaded = model_load(&loaded_tokenizer, path);

    expect(loaded != NULL && loaded_tokenizer != NULL,
           "TAGC checkpoint loads");
    expect(loaded != NULL && models_equal(fixture.model, loaded),
           "checkpoint round-trip preserves every weight bit");

    if (loaded != NULL)
        model_free(loaded);
    if (loaded_tokenizer != NULL)
        tokenizer_free(loaded_tokenizer);
}

static int corrupt_checkpoint_payload(const char *path)
{
    FILE *stream = fopen(path, "r+b");

    expect(stream != NULL, "checkpoint opens for corruption fixture");
    if (stream == NULL)
        return -1;
    expect(fseek(stream, CORRUPTION_PAYLOAD_OFFSET, SEEK_SET) == 0,
           "corruption fixture reaches payload");
    int byte = fgetc(stream);
    int corrupted =
        byte != EOF && fseek(stream, PREVIOUS_BYTE_OFFSET, SEEK_CUR) == 0
        && fputc(byte ^ LOW_BIT_MASK, stream) != EOF;

    expect(corrupted, "corruption fixture flips one finite bit");
    expect(fclose(stream) == 0, "corruption fixture closes");
    return corrupted ? 0 : -1;
}

static void check_corruption_rejection(const char *path)
{
    if (corrupt_checkpoint_payload(path) != 0)
        return;

    Tokenizer *tokenizer = NULL;
    Model *model = model_load(&tokenizer, path);

    expect(model == NULL && tokenizer == NULL,
           "CRC32 rejects finite bit corruption");
    if (model != NULL)
        model_free(model);
    if (tokenizer != NULL)
        tokenizer_free(tokenizer);
}

static void check_legacy_rejection(CheckpointFixture fixture,
                                   const char *path)
{
    expect(model_save(fixture.model, fixture.tokenizer, path) == 0,
           "clean checkpoint can replace the corrupt fixture");
    FILE *stream = fopen(path, "r+b");

    expect(stream != NULL, "legacy fixture opens");
    if (stream != NULL) {
        expect(fwrite(&LEGACY_CHECKPOINT_MAGIC,
                      sizeof LEGACY_CHECKPOINT_MAGIC, 1, stream) == 1,
               "legacy fixture writes old magic");
        expect(fclose(stream) == 0, "legacy fixture closes");
    }
    expect(rewrite_checksum(path) == 0,
           "legacy fixture keeps a valid checksum");

    Tokenizer *tokenizer = NULL;
    Model *model = model_load(&tokenizer, path);

    expect(model == NULL && tokenizer == NULL,
           "runtime loader rejects the legacy checkpoint format");
    if (model != NULL)
        model_free(model);
    if (tokenizer != NULL)
        tokenizer_free(tokenizer);
}

static void check_resource_rejection(CheckpointFixture fixture,
                                     const char *path)
{
    ModelConfig oversized = {
        .vocab_size = MINIMUM_MODEL_EXTENT,
        .block_size = RESOURCE_BOMB_EXTENT,
        .d_model = MINIMUM_MODEL_EXTENT,
        .head_count = MINIMUM_MODEL_EXTENT,
        .layer_count = MINIMUM_MODEL_EXTENT,
        .batch_size = RESOURCE_BOMB_EXTENT,
    };
    ModelMemory memory;

    expect(model_memory_requirements(oversized, &memory)
               && memory.total_bytes > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES,
           "memory preflight identifies a resource-bomb configuration");
    expect(write_resource_fixture(path, oversized) == 0,
           "resource fixture is complete and checksummed");

    Tokenizer *tokenizer = fixture.tokenizer;
    Model *model = model_load(&tokenizer, path);

    expect(model == NULL && tokenizer == NULL,
           "loader rejects the oversized arena before construction");
    if (model != NULL)
        model_free(model);
    if (tokenizer != NULL && tokenizer != fixture.tokenizer)
        tokenizer_free(tokenizer);
}

int main(void)
{
    CheckpointPaths paths = checkpoint_paths();
    int paths_ready = reserve_checkpoint_paths(&paths) == 0;

    expect(paths_ready,
           "checkpoint fixtures reserve paths");
    if (!paths_ready) {
        remove_checkpoint_paths(&paths);
        return EXIT_FAILURE;
    }

    CheckpointFixture fixture = checkpoint_fixture();

    check_save_boundaries(fixture, &paths);
    check_checkpoint_round_trip(fixture, paths.checkpoint);

    check_corruption_rejection(paths.checkpoint);
    check_legacy_rejection(fixture, paths.checkpoint);
    check_resource_rejection(fixture, paths.resource);

    checkpoint_fixture_free(fixture);
    remove_checkpoint_paths(&paths);

    if (failures != 0) {
        fprintf(stderr, "check-13: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-13: all %d checkpoint checks passed\n", checks);
    return EXIT_SUCCESS;
}
