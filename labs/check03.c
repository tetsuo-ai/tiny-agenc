/*
 * Chapter 3 witness: bytes become stable ids, durable vocabulary, and
 * shifted windows.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "rng.h"
#include "tokenizer.h"

static int checks;
static int failures;

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

int main(void)
{
    static const char VOCABULARY[] = "zaba\n";
    static const char EXPECTED[] = { '\n', 'a', 'b', 'z' };
    Tokenizer *tk = tokenizer_new(VOCABULARY, sizeof VOCABULARY - 1);

    expect(tokenizer_vocab_size(tk) == 4,
           "repeated bytes appear once in the vocabulary");
    for (int id = 0; id < 4; id++)
        expect(tokenizer_decode(tk, id) == EXPECTED[id],
               "token ids follow sorted byte order");

    int ids[5];
    size_t count = tokenizer_encode(tk, ids, "az?b\n", 5);

    expect(count == 4, "encoding skips an unknown byte");
    char decoded[sizeof ids / sizeof ids[0]];

    if (count <= sizeof decoded)
        for (size_t i = 0; i < count; i++)
            decoded[i] = tokenizer_decode(tk, ids[i]);
    expect(count == 4 && memcmp(decoded, "azb\n", 4) == 0,
           "known bytes survive an encode and decode round trip");

    Dataset *empty_ds = dataset_new(tk, "", 0);

    expect(empty_ds != NULL && dataset_token_count(empty_ds) == 0,
           "an empty source has a valid empty dataset representation");
    dataset_free(empty_ds);
    expect(dataset_max_text_bytes()
               <= (size_t)256 * (size_t)1024 * (size_t)1024,
           "the public source-file ceiling is at most 256 MiB");
    expect(dataset_new(tk, "", dataset_max_text_bytes() + 1) == NULL,
           "dataset construction rejects an unrepresentable token buffer");

    FILE *serialized = tmpfile();

    expect(serialized != NULL, "the tokenizer serialization fixture opens");
    if (serialized != NULL) {
        expect(tokenizer_write(tk, serialized) == 0,
               "the tokenizer writes its canonical vocabulary");
        rewind(serialized);

        Tokenizer *loaded = tokenizer_read(serialized);

        expect(loaded != NULL && tokenizer_vocab_size(loaded) == 4,
               "the canonical vocabulary loads");
        if (loaded != NULL) {
            for (int id = 0; id < 4; id++)
                expect(tokenizer_decode(loaded, id) == EXPECTED[id],
                       "serialized token ids keep their byte meaning");
            tokenizer_free(loaded);
        }
        fclose(serialized);
    }

    static const unsigned char DESCENDING[] = { 'z', 'a' };
    static const unsigned char DUPLICATE[] = { 'a', 'a' };
    static const unsigned char SHORT_BODY[] = { 'a' };
    static const unsigned char HIGH_BYTES[] = {
        0x00u, 0x7Fu, 0x80u, 0xFFu,
    };

    expect(read_vocabulary_fixture(2, DESCENDING,
                                   sizeof DESCENDING) == NULL,
           "tokenizer loading rejects descending byte order");
    expect(read_vocabulary_fixture(2, DUPLICATE,
                                   sizeof DUPLICATE) == NULL,
           "tokenizer loading rejects duplicate neighbors");
    expect(read_vocabulary_fixture(2, SHORT_BODY,
                                   sizeof SHORT_BODY) == NULL,
           "tokenizer loading rejects a short vocabulary body");
    expect(read_vocabulary_fixture(0, HIGH_BYTES, 0) == NULL,
           "tokenizer loading rejects an empty saved vocabulary");
    expect(read_vocabulary_fixture(257, HIGH_BYTES, 0) == NULL,
           "tokenizer loading rejects a count above the byte alphabet");

    Tokenizer *high = read_vocabulary_fixture(4, HIGH_BYTES,
                                              sizeof HIGH_BYTES);

    expect(high != NULL && tokenizer_vocab_size(high) == 4,
           "canonical unsigned high bytes load");
    if (high != NULL) {
        for (int id = 0; id < 4; id++)
            expect((unsigned char)tokenizer_decode(high, id)
                       == HIGH_BYTES[id],
                   "loaded high-byte ids keep unsigned order");
        tokenizer_free(high);
    }
    tokenizer_free(tk);

    static const char MINIMUM[] = "abcd";
    Tokenizer *minimum_tk =
        tokenizer_new(MINIMUM, sizeof MINIMUM - 1);
    Dataset *minimum_ds =
        dataset_new(minimum_tk, MINIMUM, sizeof MINIMUM - 1);
    Rng *minimum_rng = rng_new(7);
    int inputs[3];
    int targets[3];

    dataset_batch(minimum_ds, minimum_rng, inputs, targets, 1, 3);
    for (int i = 0; i < 3; i++) {
        expect(inputs[i] == i, "the only legal input window is selected");
        expect(targets[i] == i + 1, "targets are inputs shifted by one");
    }
    rng_free(minimum_rng);
    dataset_free(minimum_ds);
    tokenizer_free(minimum_tk);

    static const char DIGITS[] = "0123456789";
    enum { ROWS = 128, BLOCK = 3 };
    Tokenizer *digits_tk = tokenizer_new(DIGITS, sizeof DIGITS - 1);
    Dataset *digits_ds = dataset_new(digits_tk, DIGITS, sizeof DIGITS - 1);
    Rng *digits_rng = rng_new(42);
    int many_inputs[ROWS * BLOCK];
    int many_targets[ROWS * BLOCK];
    int saw_first = 0;
    int saw_last = 0;

    expect(dataset_token_count(digits_ds) == sizeof DIGITS - 1,
           "dataset reports its encoded token count");
    dataset_batch(digits_ds, digits_rng, many_inputs, many_targets,
                  ROWS, BLOCK);
    for (int row = 0; row < ROWS; row++) {
        int start = many_inputs[row * BLOCK];

        expect(start >= 0 && start <= 6, "batch start is inside legal bounds");
        saw_first |= start == 0;
        saw_last |= start == 6;
        for (int t = 0; t < BLOCK; t++) {
            expect(many_inputs[row * BLOCK + t] == start + t,
                   "an input row is consecutive");
            expect(many_targets[row * BLOCK + t] == start + t + 1,
                   "a target row is shifted once");
        }
    }
    expect(saw_first, "the first legal window can be sampled");
    expect(saw_last, "the final legal window can be sampled");

    rng_free(digits_rng);
    dataset_free(digits_ds);
    tokenizer_free(digits_tk);

    if (failures != 0) {
        fprintf(stderr, "check-03: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-03: all %d data checks passed\n", checks);
    return EXIT_SUCCESS;
}
