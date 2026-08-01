/* Chapter 4 witness: Mat is a row-major, non-owning view. */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "mat.h"

static int checks;
static int failures;
static float death_storage[12];

static void expect(int condition, const char *message);
static void row_below_zero(void);
static void row_at_end(void);
static void view_below_zero(void);
static void view_past_end(void);
static int aborts(void (*operation)(void));
int main(void);

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-04: %s\n", message);
    failures++;
}

static void row_below_zero(void)
{
    (void)mat_row(mat_make(death_storage, 3, 4), -1);
}

static void row_at_end(void)
{
    (void)mat_row(mat_make(death_storage, 3, 4), 3);
}

static void view_below_zero(void)
{
    (void)mat_first_rows(mat_make(death_storage, 3, 4), -1);
}

static void view_past_end(void)
{
    (void)mat_first_rows(mat_make(death_storage, 3, 4), 4);
}

static int aborts(void (*operation)(void))
{
    pid_t child = fork();

    if (child < 0) {
        perror("check-04: fork");
        exit(EXIT_FAILURE);
    }
    if (child == 0) {
        const struct rlimit no_core = { 0, 0 };

        (void)setrlimit(RLIMIT_CORE, &no_core);
        (void)close(STDERR_FILENO);
        operation();
        _exit(EXIT_SUCCESS);
    }

    int status;

    if (waitpid(child, &status, 0) < 0) {
        perror("check-04: waitpid");
        exit(EXIT_FAILURE);
    }
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

int main(void)
{
    float storage[12];

    for (int i = 0; i < 12; i++)
        storage[i] = (float)i;

    Mat full = mat_make(storage, 3, 4);

    expect(full.vals == storage, "mat_make keeps the supplied storage");
    expect(full.rows == 3 && full.cols == 4, "mat_make keeps the shape");
    expect(mat_size(full) == 12, "mat_size multiplies rows and columns");
    expect(mat_row(full, 0) == storage, "row zero starts at the buffer");
    expect(mat_row(full, 2) == storage + 8, "rows use row-major offsets");

    Mat prefix = mat_first_rows(full, 2);

    expect(prefix.vals == storage, "a leading-row view shares storage");
    expect(prefix.rows == 2 && prefix.cols == 4,
           "a leading-row view changes only its row count");
    expect(mat_size(prefix) == 8, "a view reports its visible size");

    if (prefix.vals == storage && prefix.rows == 2 && prefix.cols == 4) {
        mat_row(prefix, 1)[2] = 99.0f;
        expect(storage[6] == 99.0f, "writes through a view reach the source");
    } else {
        expect(0, "writes through a view reach the source");
    }

    expect(aborts(row_below_zero), "mat_row asserts for a negative row");
    expect(aborts(row_at_end), "mat_row asserts at the row count");
    expect(aborts(view_below_zero),
           "mat_first_rows asserts for a negative row count");
    expect(aborts(view_past_end),
           "mat_first_rows asserts above the source row count");

    if (failures != 0) {
        fprintf(stderr, "check-04: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-04: all %d matrix-view checks passed\n", checks);
    return EXIT_SUCCESS;
}
