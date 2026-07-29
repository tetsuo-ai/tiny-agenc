/*
 * Chapter 13 witness: the new TAGC checkpoint is exact, checksummed,
 * resource-bounded, and intentionally incompatible with the old magic.
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
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
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
    unsigned char buffer[4096];
    uint32_t crc = 0xFFFFFFFFu;

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
         + (size_t)config.layer_count * (12u * width * width + 4u * width)
         + 2u * width;
}

static int write_resource_fixture(const char *path, ModelConfig config)
{
    int32_t header[] = {
        (int32_t)0x43474154u,
        1,
        config.vocab_size,
        config.block_size,
        config.d_model,
        config.head_count,
        config.layer_count,
        config.batch_size,
        1,
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

int main(void)
{
    static const char TEXT[] = "\nabcabcabc\n";
    char path[] = "/tmp/tiny-agenc-check13-XXXXXX";
    char resource_path[] = "/tmp/tiny-agenc-check13-resource-XXXXXX";
    int descriptor = mkstemp(path);
    int resource_descriptor = mkstemp(resource_path);

    expect(descriptor >= 0 && resource_descriptor >= 0,
           "checkpoint fixtures reserve paths");
    if (descriptor >= 0)
        close(descriptor);
    if (resource_descriptor >= 0)
        close(resource_descriptor);

    Tokenizer *tokenizer = tokenizer_new(TEXT, sizeof TEXT - 1);
    ModelConfig config = { 4, 4, 8, 2, 1, 2 };
    Model *model = model_new(config, 91);

    expect(model_save(model, tokenizer, path) == 0,
           "TAGC checkpoint saves atomically");

    Tokenizer *loaded_tokenizer = NULL;
    Model *loaded = model_load(&loaded_tokenizer, path);

    expect(loaded != NULL && loaded_tokenizer != NULL,
           "TAGC checkpoint loads");
    expect(loaded != NULL && models_equal(model, loaded),
           "checkpoint round-trip preserves every weight bit");

    FILE *stream = fopen(path, "r+b");

    expect(stream != NULL, "checkpoint opens for corruption fixture");
    if (stream != NULL) {
        expect(fseek(stream, 40, SEEK_SET) == 0,
               "corruption fixture reaches payload");
        int byte = fgetc(stream);

        expect(byte != EOF && fseek(stream, -1, SEEK_CUR) == 0
               && fputc(byte ^ 1, stream) != EOF,
               "corruption fixture flips one finite bit");
        expect(fclose(stream) == 0, "corruption fixture closes");
    }

    Tokenizer *corrupt_tokenizer = NULL;
    Model *corrupt = model_load(&corrupt_tokenizer, path);

    expect(corrupt == NULL && corrupt_tokenizer == NULL,
           "CRC32 rejects finite bit corruption");

    expect(model_save(model, tokenizer, path) == 0,
           "clean checkpoint can replace the corrupt fixture");
    stream = fopen(path, "r+b");
    uint32_t legacy_magic = 0x4B524754u;

    expect(stream != NULL, "legacy fixture opens");
    if (stream != NULL) {
        expect(fwrite(&legacy_magic, sizeof legacy_magic, 1, stream) == 1,
               "legacy fixture writes old magic");
        expect(fclose(stream) == 0, "legacy fixture closes");
    }
    expect(rewrite_checksum(path) == 0,
           "legacy fixture keeps a valid checksum");
    Tokenizer *legacy_tokenizer = NULL;
    Model *legacy = model_load(&legacy_tokenizer, path);

    expect(legacy == NULL && legacy_tokenizer == NULL,
           "runtime loader rejects the legacy checkpoint format");

    ModelConfig oversized = {
        1, 1024, 1, 1, 1, 1024,
    };
    ModelMemory memory;

    expect(model_memory_requirements(oversized, &memory)
           && memory.total_bytes > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES,
           "memory preflight identifies a resource-bomb configuration");
    expect(write_resource_fixture(resource_path, oversized) == 0,
           "resource fixture is complete and checksummed");

    Tokenizer *resource_tokenizer = NULL;
    Model *resource_model = model_load(&resource_tokenizer, resource_path);

    expect(resource_model == NULL && resource_tokenizer == NULL,
           "loader rejects the oversized arena before construction");

    if (legacy != NULL)
        model_free(legacy);
    if (legacy_tokenizer != NULL)
        tokenizer_free(legacy_tokenizer);
    if (resource_model != NULL)
        model_free(resource_model);
    if (resource_tokenizer != NULL)
        tokenizer_free(resource_tokenizer);
    if (corrupt != NULL)
        model_free(corrupt);
    if (corrupt_tokenizer != NULL)
        tokenizer_free(corrupt_tokenizer);
    if (loaded != NULL)
        model_free(loaded);
    if (loaded_tokenizer != NULL)
        tokenizer_free(loaded_tokenizer);
    model_free(model);
    tokenizer_free(tokenizer);
    remove(path);
    remove(resource_path);

    if (failures != 0) {
        fprintf(stderr, "check-13: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-13: all %d checkpoint checks passed\n", checks);
    return EXIT_SUCCESS;
}
