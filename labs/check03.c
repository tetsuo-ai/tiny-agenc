/*
 * Chapter 3 witness: bytes become stable ids, durable vocabulary, and
 * shifted windows.
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "rng.h"
#include "tokenizer.h"

static int checks;
static int failures;

enum {
    BYTE_VALUE_COUNT = UCHAR_MAX + 1,
    BYTES_PER_MEBIBYTE = 1024 * 1024,
    MAX_EXPECTED_CORPUS_MEBIBYTES = 256,
    NEXT_TOKEN_OFFSET = 1,
    MINIMUM_DATASET_SEED = 7,
    WINDOW_SAMPLING_SEED = 42,
};

static void expect(int condition, const char *message);
static Tokenizer *read_vocabulary_fixture(int32_t count,
                                          const unsigned char *bytes,
                                          size_t stored);
static void check_tokenizer_mapping(const Tokenizer *tokenizer);
static void check_empty_dataset(const Tokenizer *tokenizer);
static void check_tokenizer_serialization(const Tokenizer *tokenizer);
static void check_saved_vocabulary_rejections(void);
static void check_minimum_dataset(void);
static void check_sampled_dataset_windows(void);
int main(void);

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-03: %s\n", message);
    failures++;
}

static Tokenizer *read_vocabulary_fixture(int32_t count,
                                          const unsigned char *bytes,
                                          size_t stored)
{
    FILE *stream = tmpfile();

    expect(stream != NULL, "the vocabulary fixture opens");
    if (stream == NULL)
        return NULL;
    int written = fwrite(&count, sizeof count, 1, stream) == 1
               && fwrite(bytes, 1, stored, stream) == stored;

    expect(written, "the vocabulary fixture is written");
    if (!written || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return NULL;
    }

    Tokenizer *tk = tokenizer_read(stream);

    expect(fclose(stream) == 0, "the vocabulary fixture closes");
    return tk;
}

static void check_tokenizer_mapping(const Tokenizer *tokenizer)
{
    static const char EXPECTED[] = { '\n', 'a', 'b', 'z' };
    static const char TEXT[] = "az?b\n";

    expect(tokenizer_vocab_size(tokenizer) == (int)sizeof EXPECTED,
           "repeated bytes appear once in the vocabulary");
    for (size_t id = 0; id < sizeof EXPECTED; id++)
        expect(tokenizer_decode(tokenizer, (int)id) == EXPECTED[id],
               "token ids follow sorted byte order");

    int ids[sizeof TEXT - 1];
    size_t count = tokenizer_encode(
        tokenizer, ids, TEXT, sizeof TEXT - 1);

    expect(count == sizeof "azb\n" - 1,
           "encoding skips an unknown byte");
    char decoded[sizeof ids / sizeof ids[0]];

    if (count <= sizeof decoded)
        for (size_t i = 0; i < count; i++)
            decoded[i] = tokenizer_decode(tokenizer, ids[i]);
    expect(count == sizeof "azb\n" - 1
               && memcmp(decoded, "azb\n", sizeof "azb\n" - 1) == 0,
           "known bytes survive an encode and decode round trip");
}

static void check_empty_dataset(const Tokenizer *tokenizer)
{
    Dataset *empty = dataset_new(tokenizer, "", 0);

    expect(empty != NULL && dataset_token_count(empty) == 0,
           "an empty source has a valid empty dataset representation");
    dataset_free(empty);

    size_t source_limit = dataset_max_text_bytes();

    expect(source_limit
               <= (size_t)MAX_EXPECTED_CORPUS_MEBIBYTES * BYTES_PER_MEBIBYTE,
           "the public source-file ceiling is at most 256 MiB");
    expect(dataset_new(tokenizer, "", source_limit + 1) == NULL,
           "dataset construction rejects an unrepresentable token buffer");
}

static void check_tokenizer_serialization(const Tokenizer *tokenizer)
{
    FILE *serialized = tmpfile();

    expect(serialized != NULL,
           "the tokenizer serialization fixture opens");
    if (serialized == NULL)
        return;

    expect(tokenizer_write(tokenizer, serialized) == 0,
           "the tokenizer writes its canonical vocabulary");
    rewind(serialized);

    Tokenizer *loaded = tokenizer_read(serialized);

    expect(loaded != NULL
               && tokenizer_vocab_size(loaded)
                      == tokenizer_vocab_size(tokenizer),
           "the canonical vocabulary loads");
    if (loaded != NULL) {
        for (int id = 0; id < tokenizer_vocab_size(tokenizer); id++)
            expect(tokenizer_decode(loaded, id)
                       == tokenizer_decode(tokenizer, id),
                   "serialized token ids keep their byte meaning");
        tokenizer_free(loaded);
    }
    fclose(serialized);
}

static void check_saved_vocabulary_rejections(void)
{
    static const unsigned char DESCENDING[] = { 'z', 'a' };
    static const unsigned char DUPLICATE[] = { 'a', 'a' };
    static const unsigned char SHORT_BODY[] = { 'a' };
    static const unsigned char HIGH_BYTES[] = {
        0x00u, 0x7Fu, 0x80u, 0xFFu,
    };

    expect(read_vocabulary_fixture((int32_t)sizeof DESCENDING,
                                   DESCENDING, sizeof DESCENDING) == NULL,
           "tokenizer loading rejects descending byte order");
    expect(read_vocabulary_fixture((int32_t)sizeof DUPLICATE,
                                   DUPLICATE, sizeof DUPLICATE) == NULL,
           "tokenizer loading rejects duplicate neighbors");
    expect(read_vocabulary_fixture((int32_t)sizeof DESCENDING,
                                   SHORT_BODY, sizeof SHORT_BODY) == NULL,
           "tokenizer loading rejects a short vocabulary body");
    expect(read_vocabulary_fixture(0, HIGH_BYTES, 0) == NULL,
           "tokenizer loading rejects an empty saved vocabulary");
    expect(read_vocabulary_fixture(BYTE_VALUE_COUNT + 1,
                                   HIGH_BYTES, 0) == NULL,
           "tokenizer loading rejects a count above the byte alphabet");

    Tokenizer *high = read_vocabulary_fixture(
        (int32_t)sizeof HIGH_BYTES, HIGH_BYTES, sizeof HIGH_BYTES);

    expect(high != NULL
               && tokenizer_vocab_size(high) == (int)sizeof HIGH_BYTES,
           "canonical unsigned high bytes load");
    if (high != NULL) {
        for (size_t id = 0; id < sizeof HIGH_BYTES; id++)
            expect((unsigned char)tokenizer_decode(high, (int)id)
                       == HIGH_BYTES[id],
                   "loaded high-byte ids keep unsigned order");
        tokenizer_free(high);
    }
}

static void check_minimum_dataset(void)
{
    static const char TEXT[] = "abcd";
    enum { BLOCK_SIZE = 3 };
    Tokenizer *tokenizer = tokenizer_new(TEXT, sizeof TEXT - 1);
    Dataset *dataset = dataset_new(tokenizer, TEXT, sizeof TEXT - 1);
    Rng *rng = rng_new(MINIMUM_DATASET_SEED);
    int inputs[BLOCK_SIZE];
    int targets[BLOCK_SIZE];

    dataset_batch(dataset, rng, inputs, targets, 1, BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i++) {
        expect(inputs[i] == i, "the only legal input window is selected");
        expect(targets[i] == i + NEXT_TOKEN_OFFSET,
               "targets are inputs shifted by one");
    }
    rng_free(rng);
    dataset_free(dataset);
    tokenizer_free(tokenizer);
}

static void check_sampled_dataset_windows(void)
{
    static const char DIGITS[] = "0123456789";
    enum {
        ROWS = 128,
        BLOCK = 3,
        LAST_START = sizeof DIGITS - 1 - BLOCK - NEXT_TOKEN_OFFSET,
    };
    Tokenizer *tokenizer = tokenizer_new(DIGITS, sizeof DIGITS - 1);
    Dataset *dataset = dataset_new(tokenizer, DIGITS, sizeof DIGITS - 1);
    Rng *rng = rng_new(WINDOW_SAMPLING_SEED);
    int inputs[ROWS * BLOCK];
    int targets[ROWS * BLOCK];
    int saw_first = 0;
    int saw_last = 0;

    expect(dataset_token_count(dataset) == sizeof DIGITS - 1,
           "dataset reports its encoded token count");
    dataset_batch(dataset, rng, inputs, targets, ROWS, BLOCK);
    for (int row = 0; row < ROWS; row++) {
        int start = inputs[row * BLOCK];

        expect(start >= 0 && start <= LAST_START,
               "batch start is inside legal bounds");
        saw_first |= start == 0;
        saw_last |= start == LAST_START;
        for (int time = 0; time < BLOCK; time++) {
            expect(inputs[row * BLOCK + time] == start + time,
                   "an input row is consecutive");
            expect(targets[row * BLOCK + time]
                       == start + time + NEXT_TOKEN_OFFSET,
                   "a target row is shifted once");
        }
    }
    expect(saw_first, "the first legal window can be sampled");
    expect(saw_last, "the final legal window can be sampled");

    rng_free(rng);
    dataset_free(dataset);
    tokenizer_free(tokenizer);
}

int main(void)
{
    static const char VOCABULARY[] = "zaba\n";
    Tokenizer *tokenizer = tokenizer_new(
        VOCABULARY, sizeof VOCABULARY - 1);

    check_tokenizer_mapping(tokenizer);
    check_empty_dataset(tokenizer);
    check_tokenizer_serialization(tokenizer);
    check_saved_vocabulary_rejections();
    tokenizer_free(tokenizer);

    check_minimum_dataset();
    check_sampled_dataset_windows();

    if (failures != 0) {
        fprintf(stderr, "check-03: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-03: all %d data checks passed\n", checks);
    return EXIT_SUCCESS;
}
