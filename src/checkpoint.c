#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "model_internal.h"
#include "util.h"

static const uint32_t CHECKPOINT_MAGIC   = 0x43474154u; /* "TAGC" */
static const int32_t  CHECKPOINT_VERSION = 1;

enum {
    CHECKPOINT_HEADER_I32S = 8,
    CHECKPOINT_TOKENIZER_SIZE_I32S = 1,
    CHECKPOINT_CHECKSUM_I32S = 1,
};

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

static int crc32_prefix(FILE *stream, off_t length, uint32_t *result)
{
    unsigned char buffer[8192];
    uint32_t      crc = 0xFFFFFFFFu;

    if (fseeko(stream, 0, SEEK_SET) != 0)
        return -1;
    while (length > 0) {
        size_t wanted =
            length < (off_t)sizeof buffer ? (size_t)length : sizeof buffer;
        size_t received = fread(buffer, 1, wanted, stream);

        if (received != wanted)
            return -1;
        crc = crc32_update(crc, buffer, received);
        length -= (off_t)received;
    }
    *result = ~crc;
    return 0;
}

static int append_checksum(FILE *stream)
{
    if (fflush(stream) != 0)
        return -1;

    off_t payload_end = ftello(stream);
    uint32_t checksum;

    if (payload_end < 0 || crc32_prefix(stream, payload_end, &checksum) != 0
        || fseeko(stream, payload_end, SEEK_SET) != 0
        || write_i32(stream, (int32_t)checksum) != 0
        || fflush(stream) != 0)
        return -1;
    return fsync(fileno(stream));
}

static int checksum_matches(FILE *stream)
{
    if (fseeko(stream, 0, SEEK_END) != 0)
        return 0;

    off_t end = ftello(stream);

    if (end < (off_t)(3 * sizeof(int32_t))
        || end > (off_t)MODEL_MAX_CHECKPOINT_FILE_BYTES)
        return 0;

    off_t   checksum_at = end - (off_t)sizeof(int32_t);
    int32_t stored;
    uint32_t computed;

    if (fseeko(stream, checksum_at, SEEK_SET) != 0
        || read_i32(stream, &stored) != 0
        || crc32_prefix(stream, checksum_at, &computed) != 0
        || (uint32_t)stored != computed)
        return 0;
    return fseeko(stream, 0, SEEK_SET) == 0;
}

static int checkpoint_config_supported(ModelConfig cfg)
{
    ModelMemory memory;
    size_t parameter_floats = model_parameter_float_count(cfg);
    size_t fixed_bytes =
        (CHECKPOINT_HEADER_I32S + CHECKPOINT_TOKENIZER_SIZE_I32S
         + CHECKPOINT_CHECKSUM_I32S) * sizeof(int32_t);

    if (!model_memory_requirements(cfg, &memory)
        || memory.total_bytes > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES
        || parameter_floats > SIZE_MAX / sizeof(float))
        return 0;

    size_t parameter_bytes = parameter_floats * sizeof(float);

    return fixed_bytes <= MODEL_MAX_CHECKPOINT_FILE_BYTES
        && (size_t)cfg.vocab_size
           <= MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
        && parameter_bytes
           <= MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
            - (size_t)cfg.vocab_size;
}

int model_save(const Model *m, const Tokenizer *tk, const char *path)
{
    static const char TEMP_SUFFIX[] = ".tmp.XXXXXX";
    size_t path_length = strlen(path);

    if (!checkpoint_config_supported(m->cfg)
        || tokenizer_vocab_size(tk) != m->cfg.vocab_size
        || path_length > SIZE_MAX - sizeof TEMP_SUFFIX)
        return -1;

    char *temp_path = emalloc(path_length + sizeof TEMP_SUFFIX);

    memcpy(temp_path, path, path_length);
    memcpy(temp_path + path_length, TEMP_SUFFIX, sizeof TEMP_SUFFIX);

    int descriptor = mkstemp(temp_path);

    if (descriptor < 0) {
        free(temp_path);
        return -1;
    }

    struct stat existing;

    if (stat(path, &existing) == 0
        && fchmod(descriptor, existing.st_mode & 0777) != 0) {
        close(descriptor);
        remove(temp_path);
        free(temp_path);
        return -1;
    }

    FILE *stream = fdopen(descriptor, "w+b");

    if (stream == NULL) {
        close(descriptor);
        remove(temp_path);
        free(temp_path);
        return -1;
    }

    int failed = write_i32(stream, (int32_t)CHECKPOINT_MAGIC) != 0
              || write_i32(stream, CHECKPOINT_VERSION) != 0
              || write_i32(stream, m->cfg.vocab_size) != 0
              || write_i32(stream, m->cfg.block_size) != 0
              || write_i32(stream, m->cfg.d_model) != 0
              || write_i32(stream, m->cfg.head_count) != 0
              || write_i32(stream, m->cfg.layer_count) != 0
              || write_i32(stream, m->cfg.batch_size) != 0
              || tokenizer_write(tk, stream) != 0;

    for (int i = 0; !failed && i < m->param_count; i++)
        failed = param_write(m->params[i], stream) != 0;
    if (!failed)
        failed = append_checksum(stream) != 0;
    if (fclose(stream) != 0)
        failed = 1;
    if (!failed && rename(temp_path, path) != 0)
        failed = 1;
    if (failed)
        remove(temp_path);
    free(temp_path);
    return failed ? -1 : 0;
}

static int read_dimension(FILE *stream, int *value)
{
    int32_t raw;

    if (read_i32(stream, &raw) != 0)
        return -1;
    *value = (int)raw;
    return 0;
}

static int read_config(FILE *stream, ModelConfig *cfg)
{
    int32_t magic;
    int32_t version;

    if (read_i32(stream, &magic) != 0
        || (uint32_t)magic != CHECKPOINT_MAGIC
        || read_i32(stream, &version) != 0
        || version != CHECKPOINT_VERSION)
        return -1;
    if (read_dimension(stream, &cfg->vocab_size) != 0
        || read_dimension(stream, &cfg->block_size) != 0
        || read_dimension(stream, &cfg->d_model) != 0
        || read_dimension(stream, &cfg->head_count) != 0
        || read_dimension(stream, &cfg->layer_count) != 0
        || read_dimension(stream, &cfg->batch_size) != 0)
        return -1;
    return model_config_valid(*cfg) ? 0 : -1;
}

static int payload_size_matches(FILE *stream, ModelConfig cfg)
{
    off_t here = ftello(stream);

    if (here < 0 || fseeko(stream, 0, SEEK_END) != 0)
        return 0;

    off_t end = ftello(stream);
    size_t parameter_floats = model_parameter_float_count(cfg);

    if (end < here || end - here < (off_t)sizeof(int32_t)
        || parameter_floats > SIZE_MAX / sizeof(float)
        || fseeko(stream, here, SEEK_SET) != 0)
        return 0;
    return (uintmax_t)(end - here - (off_t)sizeof(int32_t))
        == (uintmax_t)parameter_floats * sizeof(float);
}

Model *model_load(Tokenizer **tk, const char *path)
{
    if (tk == NULL)
        return NULL;
    *tk = NULL;

    FILE *stream = fopen(path, "rb");
    ModelConfig cfg;
    if (stream == NULL)
        return NULL;
    if (!checksum_matches(stream)
        || read_config(stream, &cfg) != 0
        || !checkpoint_config_supported(cfg)) {
        fclose(stream);
        return NULL;
    }

    Tokenizer *loaded_tk = tokenizer_read(stream);

    if (loaded_tk == NULL
        || tokenizer_vocab_size(loaded_tk) != cfg.vocab_size) {
        if (loaded_tk != NULL)
            tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    if (!payload_size_matches(stream, cfg)) {
        tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    Model *m = model_new(cfg, 0);

    if (model_parameter_count(m) != model_parameter_float_count(cfg)) {
        model_free(m);
        tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    for (int i = 0; i < m->param_count; i++) {
        if (param_read(m->params[i], stream) != 0) {
            model_free(m);
            tokenizer_free(loaded_tk);
            fclose(stream);
            return NULL;
        }
    }

    int32_t checksum;

    if (read_i32(stream, &checksum) != 0 || fgetc(stream) != EOF) {
        model_free(m);
        tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    fclose(stream);
    *tk = loaded_tk;
    return m;
}
