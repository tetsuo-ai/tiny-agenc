#include <assert.h>
#include <stdlib.h>

#include "tokenizer.h"
#include "util.h"

enum {
    BYTE_VALUES  = 256,
    TOKEN_ABSENT = -1,
};

struct Tokenizer {
    int  vocab_size;
    char id_to_byte[BYTE_VALUES];
    int  byte_to_id[BYTE_VALUES];
};

/* Walking byte values in ascending order sorts the vocabulary. */
static Tokenizer *from_seen_bytes(const int seen[BYTE_VALUES])
{
    Tokenizer *tk = emalloc(sizeof *tk);

    tk->vocab_size = 0;
    for (int byte = 0; byte < BYTE_VALUES; byte++) {
        tk->byte_to_id[byte] = TOKEN_ABSENT;
        if (!seen[byte])
            continue;
        tk->byte_to_id[byte] = tk->vocab_size;
        tk->id_to_byte[tk->vocab_size] = (char)byte;
        tk->vocab_size++;
    }
    return tk;
}

Tokenizer *tokenizer_new(const char *text, size_t length)
{
    int seen[BYTE_VALUES] = { 0 };

    for (size_t i = 0; i < length; i++)
        seen[(unsigned char)text[i]] = 1;
    return from_seen_bytes(seen);
}

int tokenizer_vocab_size(const Tokenizer *tk)
{
    return tk->vocab_size;
}

size_t tokenizer_encode(const Tokenizer *tk, int *ids, const char *text, size_t length)
{
    size_t written = 0;

    for (size_t i = 0; i < length; i++) {
        int id = tk->byte_to_id[(unsigned char)text[i]];

        if (id == TOKEN_ABSENT)
            continue;
        ids[written++] = id;
    }
    return written;
}

char tokenizer_decode(const Tokenizer *tk, int id)
{
    assert(id >= 0 && id < tk->vocab_size);
    return tk->id_to_byte[id];
}

int tokenizer_write(const Tokenizer *tk, FILE *stream)
{
    if (write_i32(stream, tk->vocab_size) != 0)
        return -1;
    if (fwrite(tk->id_to_byte, 1, (size_t)tk->vocab_size, stream) != (size_t)tk->vocab_size)
        return -1;
    return 0;
}

static int read_vocabulary_size(FILE *stream, int32_t *vocab_size)
{
    if (read_i32(stream, vocab_size) != 0)
        return -1;
    if (*vocab_size < 1 || *vocab_size > BYTE_VALUES)
        return -1;
    return 0;
}

static int read_vocabulary_bytes(FILE *stream, char bytes[BYTE_VALUES],
                                 int32_t vocab_size)
{
    return fread(bytes, 1, (size_t)vocab_size, stream)
               == (size_t)vocab_size
        ? 0 : -1;
}

static int vocabulary_is_canonical(const char bytes[BYTE_VALUES],
                                   int32_t vocab_size)
{
    for (int32_t i = 1; i < vocab_size; i++) {
        /* Weight rows use this serialized id order.  The writer emits
         * ascending bytes, so accepting any other order would silently
         * attach the loaded weights to different characters. */
        if ((unsigned char)bytes[i - 1] >= (unsigned char)bytes[i])
            return 0;
    }
    return 1;
}

static Tokenizer *from_canonical_vocabulary(const char bytes[BYTE_VALUES],
                                            int32_t vocab_size)
{
    int seen[BYTE_VALUES] = { 0 };

    for (int32_t i = 0; i < vocab_size; i++)
        seen[(unsigned char)bytes[i]] = 1;
    return from_seen_bytes(seen);
}

Tokenizer *tokenizer_read(FILE *stream)
{
    int32_t vocab_size;

    if (read_vocabulary_size(stream, &vocab_size) != 0)
        return NULL;

    char bytes[BYTE_VALUES];

    if (read_vocabulary_bytes(stream, bytes, vocab_size) != 0)
        return NULL;
    if (!vocabulary_is_canonical(bytes, vocab_size))
        return NULL;
    return from_canonical_vocabulary(bytes, vocab_size);
}

void tokenizer_free(Tokenizer *tk)
{
    free(tk);
}
