/*
 * tokenizer.h -- from characters to integers and back.
 *
 * A language model works on integer token ids.  Tiny AgenC's tokens are
 * single bytes: the vocabulary is every distinct character in the
 * training corpus, sorted so that ids are stable across runs.  Larger
 * language models commonly use learned subword vocabularies for the
 * same byte-or-text-to-id role.
 */
#ifndef TINY_AGENC_TOKENIZER_H
#define TINY_AGENC_TOKENIZER_H

#include <stddef.h>
#include <stdio.h>

typedef struct Tokenizer Tokenizer;

Tokenizer *tokenizer_new(const char *text, size_t length);
int        tokenizer_vocab_size(const Tokenizer *tk);

/* Encoding skips bytes outside the vocabulary and returns the number
 * of ids written; ids is at least as large as text.  Decoding requires
 * a valid id from this tokenizer's vocabulary. */
size_t     tokenizer_encode(const Tokenizer *tk, int *ids, const char *text, size_t length);
char       tokenizer_decode(const Tokenizer *tk, int id);

/* Checkpoint I/O: write returns 0 on success, read returns NULL on a
 * malformed stream. */
int        tokenizer_write(const Tokenizer *tk, FILE *stream);
Tokenizer *tokenizer_read(FILE *stream);

void       tokenizer_free(Tokenizer *tk);

#endif
