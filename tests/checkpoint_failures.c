#define _GNU_SOURCE

#include <acl/libacl.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>

#include "model_internal.h"
#include "tokenizer.h"
#include "util.h"

typedef enum {
    INJECT_NONE,
    INJECT_FILE_FSYNC,
    INJECT_DIRECTORY_PREFLIGHT_FSYNC,
    INJECT_DIRECTORY_COMMIT_FSYNC,
    INJECT_DIRECTORY_CLOSE,
    INJECT_FCHOWN,
    INJECT_FCHMOD,
    INJECT_ACL_SET,
    INJECT_METADATA_MISMATCH,
    INJECT_XATTR_SET,
    INJECT_FFLUSH,
    INJECT_FCLOSE,
    INJECT_RENAME,
    INJECT_FMEMOPEN,
    MUTATE_AT_FMEMOPEN,
    MUTATE_TEMP_AFTER_CLOSE,
} Injection;

typedef struct {
    Model     *model;
    Tokenizer *tokenizer;
} Fixture;

typedef struct {
    Injection      injection;
    const char    *name;
    ModelSaveResult expected;
} FailureCase;

typedef struct {
    char metadata[PATH_MAX];
    char snapshot[PATH_MAX];
    char registry[PATH_MAX];
    char identity[PATH_MAX];
    char golden[PATH_MAX];
    char fmemopen[PATH_MAX];
} CheckpointFailurePaths;

enum {
    FILE_PERMISSION_MASK = 07777,
    PRIVATE_FILE_MODE = 0600,
    TEST_METADATA_MODE = 06740,
    PRIVATE_DIRECTORY_MODE = 0700,
    FAILURE_MESSAGE_CAPACITY = 160,
    XATTR_VALUE_CAPACITY = 32,
    EXPECTED_INJECTION_HITS = 1,
    DESTINATION_AND_FOREIGN_ENTRIES = 2,
    FOREIGN_ENTRY_COUNT = 1,
    BLOCK_ALL_PERMISSION_BITS = 0777,
    GOLDEN_VALUE_PERIOD = 17,
    GOLDEN_VALUE_CENTER = 8,
    GOLDEN_PARAMETER_SCALARS = 20,
};

static const unsigned long long FIXTURE_MODEL_SEED = 2026;
static const unsigned long long GOLDEN_MODEL_SEED = 1337;
static const float GOLDEN_VALUE_SCALE = 0.125f;

static const unsigned char OLD_BYTES[] = {
    'o', 'l', 'd', ' ', 'c', 'h', 'e', 'c', 'k', 'p', 'o', 'i', 'n', 't',
};
static const unsigned char BINARY_XATTR[] = { 0x00, 0x41, 0xFF, 0x00 };
static const char EMPTY_XATTR_NAME[] = "user.tiny-agenc-empty";
static const char BINARY_XATTR_NAME[] = "user.tiny-agenc-binary";
static const size_t GOLDEN_CHECKPOINT_BYTES = 121;
/*
 * FNV-1a over the deterministic 121-byte file written by the
 * pre-refactor TAGC v1 writer at af73d2d.
 */
static const uint64_t GOLDEN_CHECKPOINT_FNV1A =
    UINT64_C(0xdf6959f32448a235);

static Injection active_injection;
static int       injection_hits;
static int       directory_fsync_calls;
static const char *mutation_path;
static const char *mutation_directory;
static int       mutation_succeeded;
static char      swapped_temp_path[PATH_MAX];
static char      displaced_temp_path[PATH_MAX];
static int       failures;
static int       checks;

extern int   __real_acl_set_fd(int descriptor, acl_t acl);
extern int   __real_acl_cmp(acl_t first, acl_t second);
extern int   __real_close(int descriptor);
extern int   __real_fchmod(int descriptor, mode_t mode);
extern int   __real_fchown(int descriptor, uid_t owner, gid_t group);
extern int   __real_fclose(FILE *stream);
extern int   __real_fflush(FILE *stream);
extern FILE *__real_fmemopen(void *buffer, size_t size, const char *mode);
extern int   __real_fsetxattr(int descriptor, const char *name,
                             const void *value, size_t size, int flags);
extern int   __real_fsync(int descriptor);
extern int   __real_renameat(int old_directory, const char *old_path,
                             int new_directory, const char *new_path);
extern int   __real_renameat2(int old_directory, const char *old_path,
                              int new_directory, const char *new_path,
                              unsigned int flags);

static int injected_failure(void);
int __wrap_acl_set_fd(int descriptor, acl_t acl);
int __wrap_acl_cmp(acl_t first, acl_t second);
int __wrap_close(int descriptor);
int __wrap_fchmod(int descriptor, mode_t mode);
int __wrap_fchown(int descriptor, uid_t owner, gid_t group);
int __wrap_fflush(FILE *stream);
int __wrap_fclose(FILE *stream);
static int overwrite_with_corruption(const char *path);
FILE *__wrap_fmemopen(void *buffer, size_t size, const char *mode);
int __wrap_fsetxattr(int descriptor, const char *name, const void *value,
                     size_t size, int flags);
int __wrap_fsync(int descriptor);
int __wrap_renameat(int old_directory, const char *old_path, int new_directory,
                    const char *new_path);
int __wrap_renameat2(int old_directory, const char *old_path,
                     int new_directory, const char *new_path,
                     unsigned int flags);
static void expect(int condition, const char *message);
static int join_path(char *result, size_t capacity, const char *directory,
                     const char *name);
static int replace_with_bytes(const char *path, const unsigned char *bytes,
                              size_t size);
static int find_temporary_entry_name(char *name, size_t capacity);
static int swap_temporary_entry(void);
static int file_equals(const char *path, const unsigned char *bytes,
                       size_t size);
static int set_test_acl(const char *path);
static int set_test_metadata(const char *path);
static int xattr_equals(const char *path, const char *name,
                        const unsigned char *expected, size_t expected_size);
static int directory_entry_count(const char *path);
static Fixture fixture_new(void);
static void fixture_free(Fixture fixture);
static int checkpoint_loads(const char *path);
static void check_metadata_round_trip(Fixture fixture, const char *path);
static void check_rejected_destinations(Fixture fixture,
                                        const char *directory);
static void start_injection(Injection injection);
static int prepare_failure_destination(const char *path);
static void check_injected_failure_case(Fixture fixture,
                                        const FailureCase *failure_case,
                                        const char *path,
                                        const char *directory,
                                        const char *foreign);
static void check_absent_rename_failure(Fixture fixture,
                                        const char *path,
                                        const char *directory,
                                        const char *foreign);
static void check_injected_save_failures(Fixture fixture,
                                         const char *directory);
static void check_immutable_load_snapshot(Fixture fixture, const char *path);
static void check_fmemopen_failure(Fixture fixture, const char *path);
static void check_registry_preflight(Fixture fixture, const char *path);
static void check_temporary_identity(Fixture fixture, const char *directory,
                                     const char *path);
static uint64_t fnv1a64(const unsigned char *bytes, size_t size);
static int golden_native_format_matches(void);
static void check_writer_golden(const char *path);
static int checkpoint_failure_paths(const char *directory,
                                    CheckpointFailurePaths *paths);
int main(void);

static int injected_failure(void)
{
    injection_hits++;
    errno = EIO;
    return -1;
}

int __wrap_acl_set_fd(int descriptor, acl_t acl)
{
    if (active_injection == INJECT_ACL_SET)
        return injected_failure();
    return __real_acl_set_fd(descriptor, acl);
}

int __wrap_acl_cmp(acl_t first, acl_t second)
{
    if (active_injection == INJECT_METADATA_MISMATCH) {
        injection_hits++;
        return 1;
    }
    return __real_acl_cmp(first, second);
}

int __wrap_close(int descriptor)
{
    struct stat status;
    int is_directory =
        fstat(descriptor, &status) == 0 && S_ISDIR(status.st_mode);
    int result = __real_close(descriptor);

    if (active_injection == INJECT_DIRECTORY_CLOSE && is_directory) {
        injection_hits++;
        errno = EIO;
        return -1;
    }
    return result;
}

int __wrap_fchmod(int descriptor, mode_t mode)
{
    if (active_injection == INJECT_FCHMOD)
        return injected_failure();
    return __real_fchmod(descriptor, mode);
}

int __wrap_fchown(int descriptor, uid_t owner, gid_t group)
{
    if (active_injection == INJECT_FCHOWN)
        return injected_failure();
    return __real_fchown(descriptor, owner, group);
}

int __wrap_fflush(FILE *stream)
{
    if (active_injection == INJECT_FFLUSH)
        return injected_failure();
    return __real_fflush(stream);
}

int __wrap_fclose(FILE *stream)
{
    int result = __real_fclose(stream);

    if (active_injection == MUTATE_TEMP_AFTER_CLOSE && result == 0) {
        injection_hits++;
        mutation_succeeded = swap_temporary_entry() == 0;
    }
    if (active_injection == INJECT_FCLOSE) {
        injection_hits++;
        errno = EIO;
        return EOF;
    }
    return result;
}

static int overwrite_with_corruption(const char *path)
{
    static const unsigned char CORRUPTION[] = { 'b', 'a', 'd' };
    int descriptor = open(path, O_WRONLY | O_TRUNC | O_CLOEXEC);

    if (descriptor < 0)
        return -1;
    ssize_t written = write(descriptor, CORRUPTION, sizeof CORRUPTION);
    int close_failed = close(descriptor) != 0;

    return written == (ssize_t)sizeof CORRUPTION && !close_failed ? 0 : -1;
}

FILE *__wrap_fmemopen(void *buffer, size_t size, const char *mode)
{
    if (active_injection == INJECT_FMEMOPEN) {
        injected_failure();
        return NULL;
    }
    if (active_injection == MUTATE_AT_FMEMOPEN) {
        injection_hits++;
        mutation_succeeded = overwrite_with_corruption(mutation_path) == 0;
    }
    return __real_fmemopen(buffer, size, mode);
}

int __wrap_fsetxattr(int descriptor, const char *name, const void *value,
                     size_t size, int flags)
{
    if (active_injection == INJECT_XATTR_SET)
        return injected_failure();
    return __real_fsetxattr(descriptor, name, value, size, flags);
}

int __wrap_fsync(int descriptor)
{
    struct stat status;
    int is_directory =
        fstat(descriptor, &status) == 0 && S_ISDIR(status.st_mode);

    if (is_directory)
        directory_fsync_calls++;
    if ((active_injection == INJECT_FILE_FSYNC && !is_directory)
        || (active_injection == INJECT_DIRECTORY_PREFLIGHT_FSYNC
            && is_directory && directory_fsync_calls == 1)
        || (active_injection == INJECT_DIRECTORY_COMMIT_FSYNC
            && is_directory && directory_fsync_calls == 2))
        return injected_failure();
    return __real_fsync(descriptor);
}

int __wrap_renameat(int old_directory, const char *old_path,
                    int new_directory, const char *new_path)
{
    if (active_injection == INJECT_RENAME)
        return injected_failure();
    return __real_renameat(old_directory, old_path, new_directory, new_path);
}

int __wrap_renameat2(int old_directory, const char *old_path,
                     int new_directory, const char *new_path,
                     unsigned int flags)
{
    if (active_injection == INJECT_RENAME)
        return injected_failure();
    return __real_renameat2(old_directory, old_path, new_directory, new_path,
                            flags);
}

static void expect(int condition, const char *message)
{
    checks++;
    if (!condition) {
        fprintf(stderr, "checkpoint failure: %s\n", message);
        failures++;
    }
}

static int join_path(char *result, size_t capacity, const char *directory,
                     const char *name)
{
    int length = snprintf(result, capacity, "%s/%s", directory, name);

    return length >= 0 && (size_t)length < capacity ? 0 : -1;
}

static int replace_with_bytes(const char *path, const unsigned char *bytes,
                              size_t size)
{
    if (unlink(path) != 0 && errno != ENOENT)
        return -1;

    int descriptor =
        open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
             PRIVATE_FILE_MODE);

    if (descriptor < 0)
        return -1;
    ssize_t written = write(descriptor, bytes, size);
    int close_failed = close(descriptor) != 0;

    return written == (ssize_t)size && !close_failed ? 0 : -1;
}

static int find_temporary_entry_name(char *name, size_t capacity)
{
    static const char PREFIX[] = ".tiny-agenc-";
    DIR *directory = opendir(mutation_directory);

    if (directory == NULL)
        return -1;

    struct dirent *entry;

    while ((entry = readdir(directory)) != NULL) {
        if (strncmp(entry->d_name, PREFIX, sizeof PREFIX - 1) != 0)
            continue;

        size_t length = strlen(entry->d_name);

        if (length >= capacity) {
            closedir(directory);
            return -1;
        }
        memcpy(name, entry->d_name, length + 1);
        break;
    }
    return closedir(directory) == 0 && name[0] != '\0' ? 0 : -1;
}

static int swap_temporary_entry(void)
{
    char name[NAME_MAX + 1] = { 0 };

    if (find_temporary_entry_name(name, sizeof name) != 0
        || join_path(swapped_temp_path, sizeof swapped_temp_path,
                     mutation_directory, name) != 0
        || join_path(displaced_temp_path, sizeof displaced_temp_path,
                     mutation_directory, ".displaced-owned-checkpoint") != 0
        || rename(swapped_temp_path, displaced_temp_path) != 0)
        return -1;
    return replace_with_bytes(swapped_temp_path, OLD_BYTES,
                              sizeof OLD_BYTES);
}

static int file_equals(const char *path, const unsigned char *bytes,
                       size_t size)
{
    unsigned char buffer[sizeof OLD_BYTES];
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);

    if (descriptor < 0 || size > sizeof buffer) {
        if (descriptor >= 0)
            close(descriptor);
        return 0;
    }
    ssize_t received = read(descriptor, buffer, size);
    unsigned char trailing;
    ssize_t extra = read(descriptor, &trailing, 1);
    int close_failed = close(descriptor) != 0;

    return received == (ssize_t)size && extra == 0 && !close_failed
        && memcmp(buffer, bytes, size) == 0;
}

static int set_test_acl(const char *path)
{
    char text[FAILURE_MESSAGE_CAPACITY];
    unsigned long named_user = (unsigned long)getuid() + 1UL;
    int length = snprintf(text, sizeof text,
                          "user::rwx,user:%lu:r--,group::r--,"
                          "mask::r--,other::---",
                          named_user);

    if (length < 0 || (size_t)length >= sizeof text)
        return -1;

    acl_t acl = acl_from_text(text);

    if (acl == NULL)
        return -1;
    int result = acl_set_file(path, ACL_TYPE_ACCESS, acl);

    acl_free(acl);
    return result;
}

static int set_test_metadata(const char *path)
{
    if (set_test_acl(path) != 0 || chmod(path, TEST_METADATA_MODE) != 0)
        return -1;
    if (setxattr(path, EMPTY_XATTR_NAME, "", 0, 0) != 0)
        return -1;
    return setxattr(path, BINARY_XATTR_NAME, BINARY_XATTR,
                    sizeof BINARY_XATTR, 0);
}

static int xattr_equals(const char *path, const char *name,
                        const unsigned char *expected, size_t expected_size)
{
    unsigned char buffer[XATTR_VALUE_CAPACITY];
    ssize_t received = getxattr(path, name, buffer, sizeof buffer);

    return received == (ssize_t)expected_size
        && memcmp(buffer, expected, expected_size) == 0;
}

static int directory_entry_count(const char *path)
{
    DIR *directory = opendir(path);

    if (directory == NULL)
        return -1;

    int count = 0;
    struct dirent *entry;

    while ((entry = readdir(directory)) != NULL) {
        if (strcmp(entry->d_name, ".") != 0
            && strcmp(entry->d_name, "..") != 0)
            count++;
    }
    if (closedir(directory) != 0)
        return -1;
    return count;
}

static Fixture fixture_new(void)
{
    static const char TEXT[] = "\nabcabcabc\n";
    Fixture fixture;

    fixture.tokenizer = tokenizer_new(TEXT, sizeof TEXT - 1);
    ModelConfig config = {
        .vocab_size  = 4,
        .block_size  = 2,
        .d_model     = 4,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = 1,
    };
    fixture.model = model_new(config, FIXTURE_MODEL_SEED);
    return fixture;
}

static void fixture_free(Fixture fixture)
{
    model_free(fixture.model);
    tokenizer_free(fixture.tokenizer);
}

static int checkpoint_loads(const char *path)
{
    Tokenizer *tokenizer = NULL;
    Model *model = model_load(&tokenizer, path);
    int loaded = model != NULL && tokenizer != NULL;

    if (model != NULL)
        model_free(model);
    if (tokenizer != NULL)
        tokenizer_free(tokenizer);
    return loaded;
}

static void check_metadata_round_trip(Fixture fixture, const char *path)
{
    mode_t previous_mask = umask(BLOCK_ALL_PERMISSION_BITS);
    ModelSaveResult created_result =
        model_save_durable(fixture.model, fixture.tokenizer, path);
    umask(previous_mask);

    expect(created_result == MODEL_SAVE_DURABLE,
           "a new checkpoint is durable under a restrictive umask");

    struct stat created;

    expect(stat(path, &created) == 0
               && (created.st_mode & FILE_PERMISSION_MASK)
                      == PRIVATE_FILE_MODE,
           "a new checkpoint has mode 0600 independent of umask");
    expect(checkpoint_loads(path),
           "the explicitly readable new checkpoint loads");
    expect(set_test_metadata(path) == 0,
           "mode, ACL, and user xattr fixtures are installed");

    struct stat before;
    acl_t expected_acl = acl_get_file(path, ACL_TYPE_ACCESS);

    expect(stat(path, &before) == 0 && expected_acl != NULL,
           "checkpoint metadata can be captured");
    expect(model_save_durable(fixture.model, fixture.tokenizer, path)
               == MODEL_SAVE_DURABLE,
           "metadata-preserving replacement is durable");

    struct stat after;
    acl_t actual_acl = acl_get_file(path, ACL_TYPE_ACCESS);

    expect(stat(path, &after) == 0
               && (after.st_mode & FILE_PERMISSION_MASK)
                      == (before.st_mode & FILE_PERMISSION_MASK),
           "checkpoint replacement preserves all mode bits");
    expect(after.st_uid == before.st_uid && after.st_gid == before.st_gid,
           "checkpoint replacement preserves owner and group");
    expect(expected_acl != NULL && actual_acl != NULL
               && acl_cmp(expected_acl, actual_acl) == 0,
           "checkpoint replacement preserves its access ACL");
    expect(xattr_equals(path, EMPTY_XATTR_NAME,
                        (const unsigned char *)"", 0),
           "checkpoint replacement preserves an empty xattr");
    expect(xattr_equals(path, BINARY_XATTR_NAME, BINARY_XATTR,
                        sizeof BINARY_XATTR),
           "checkpoint replacement preserves a binary xattr");
    expect(checkpoint_loads(path),
           "metadata-preserving replacement remains loadable");

    if (expected_acl != NULL)
        acl_free(expected_acl);
    if (actual_acl != NULL)
        acl_free(actual_acl);
}

static void check_rejected_destinations(Fixture fixture,
                                        const char *directory)
{
    char target[PATH_MAX];
    char symlink_path[PATH_MAX];
    char fifo_path[PATH_MAX];
    char directory_path[PATH_MAX];

    expect(join_path(target, sizeof target, directory, "symlink-target") == 0
               && join_path(symlink_path, sizeof symlink_path, directory,
                            "checkpoint-link") == 0
               && join_path(fifo_path, sizeof fifo_path, directory,
                            "checkpoint-fifo") == 0
               && join_path(directory_path, sizeof directory_path, directory,
                            "checkpoint-directory") == 0,
           "special destination paths fit");
    expect(replace_with_bytes(target, OLD_BYTES, sizeof OLD_BYTES) == 0
               && symlink(target, symlink_path) == 0,
           "symlink destination fixture is ready");
    expect(model_save_durable(fixture.model, fixture.tokenizer, symlink_path)
               == MODEL_SAVE_NOT_COMMITTED,
           "checkpoint save rejects a symlink destination");

    struct stat link_status;

    expect(lstat(symlink_path, &link_status) == 0
               && S_ISLNK(link_status.st_mode)
               && file_equals(target, OLD_BYTES, sizeof OLD_BYTES),
           "rejected symlink leaves its target untouched");
    expect(mkfifo(fifo_path, PRIVATE_FILE_MODE) == 0,
           "FIFO destination fixture is ready");
    expect(model_save_durable(fixture.model, fixture.tokenizer, fifo_path)
               == MODEL_SAVE_NOT_COMMITTED,
           "checkpoint save rejects a FIFO destination");

    struct stat fifo_status;

    expect(lstat(fifo_path, &fifo_status) == 0
               && S_ISFIFO(fifo_status.st_mode),
           "rejected FIFO remains a FIFO");
    expect(mkdir(directory_path, PRIVATE_DIRECTORY_MODE) == 0,
           "directory destination fixture is ready");
    expect(model_save_durable(fixture.model, fixture.tokenizer, directory_path)
               == MODEL_SAVE_NOT_COMMITTED,
           "checkpoint save rejects a directory destination");

    struct stat directory_status;

    expect(lstat(directory_path, &directory_status) == 0
               && S_ISDIR(directory_status.st_mode),
           "rejected directory remains a directory");

    unlink(symlink_path);
    unlink(fifo_path);
    unlink(target);
    rmdir(directory_path);
}

static void start_injection(Injection injection)
{
    active_injection = injection;
    injection_hits = 0;
    directory_fsync_calls = 0;
}

static int prepare_failure_destination(const char *path)
{
    return replace_with_bytes(path, OLD_BYTES, sizeof OLD_BYTES) == 0
        && set_test_metadata(path) == 0 ? 0 : -1;
}

static void check_injected_failure_case(Fixture fixture,
                                        const FailureCase *failure_case,
                                        const char *path,
                                        const char *directory,
                                        const char *foreign)
{
    expect(prepare_failure_destination(path) == 0,
           "old destination and metadata are ready");
    start_injection(failure_case->injection);
    ModelSaveResult result =
        model_save_durable(fixture.model, fixture.tokenizer, path);
    active_injection = INJECT_NONE;

    char message[FAILURE_MESSAGE_CAPACITY];

    snprintf(message, sizeof message, "%s failure reports its commit state",
             failure_case->name);
    expect(result == failure_case->expected, message);
    snprintf(message, sizeof message, "%s failure point was reached",
             failure_case->name);
    expect(injection_hits == EXPECTED_INJECTION_HITS, message);

    int committed = failure_case->expected
                 == MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED;

    snprintf(message, sizeof message,
             "%s failure leaves the correct destination bytes",
             failure_case->name);
    expect(committed ? checkpoint_loads(path)
                     : file_equals(path, OLD_BYTES, sizeof OLD_BYTES),
           message);
    snprintf(message, sizeof message,
             "%s failure removes only its owned temporary file",
             failure_case->name);
    expect(directory_entry_count(directory)
               == DESTINATION_AND_FOREIGN_ENTRIES
               && file_equals(foreign, OLD_BYTES, sizeof OLD_BYTES),
           message);
}

static void check_absent_rename_failure(Fixture fixture,
                                        const char *path,
                                        const char *directory,
                                        const char *foreign)
{
    unlink(path);
    start_injection(INJECT_RENAME);
    ModelSaveResult result =
        model_save_durable(fixture.model, fixture.tokenizer, path);
    active_injection = INJECT_NONE;

    expect(result == MODEL_SAVE_NOT_COMMITTED
               && injection_hits == EXPECTED_INJECTION_HITS,
           "no-replace rename failure reports not committed");
    expect(access(path, F_OK) != 0 && errno == ENOENT,
           "no-replace rename failure leaves no destination");
    expect(directory_entry_count(directory) == FOREIGN_ENTRY_COUNT
               && file_equals(foreign, OLD_BYTES, sizeof OLD_BYTES),
           "no-replace rename failure removes its owned temporary file");
}

static void check_injected_save_failures(Fixture fixture,
                                         const char *directory)
{
    static const FailureCase CASES[] = {
        { INJECT_FILE_FSYNC, "file fsync", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_DIRECTORY_PREFLIGHT_FSYNC, "directory preflight fsync",
          MODEL_SAVE_NOT_COMMITTED },
        { INJECT_FCHOWN, "fchown", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_FCHMOD, "fchmod", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_ACL_SET, "acl_set_fd", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_METADATA_MISMATCH, "metadata re-read",
          MODEL_SAVE_NOT_COMMITTED },
        { INJECT_XATTR_SET, "fsetxattr", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_FFLUSH, "fflush", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_FCLOSE, "fclose", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_RENAME, "rename", MODEL_SAVE_NOT_COMMITTED },
        { INJECT_DIRECTORY_COMMIT_FSYNC, "directory commit fsync",
          MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED },
        { INJECT_DIRECTORY_CLOSE, "directory close",
          MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED },
    };
    char path[PATH_MAX];
    char foreign[PATH_MAX];

    expect(join_path(path, sizeof path, directory, "failure.bin") == 0
               && join_path(foreign, sizeof foreign, directory,
                            ".tiny-agenc-00000000000000000000000000000000")
                      == 0,
           "failure-test paths fit");
    expect(replace_with_bytes(foreign, OLD_BYTES, sizeof OLD_BYTES) == 0,
           "foreign temporary-looking file is ready");

    for (size_t i = 0; i < sizeof CASES / sizeof CASES[0]; i++)
        check_injected_failure_case(fixture, &CASES[i], path,
                                    directory, foreign);

    check_absent_rename_failure(fixture, path, directory, foreign);

    unlink(foreign);
}

static void check_immutable_load_snapshot(Fixture fixture,
                                          const char *path)
{
    expect(model_save_durable(fixture.model, fixture.tokenizer, path)
               == MODEL_SAVE_DURABLE,
           "snapshot fixture saves");

    mutation_path = path;
    mutation_succeeded = 0;
    start_injection(MUTATE_AT_FMEMOPEN);

    Tokenizer *tokenizer = NULL;
    Model *model = model_load(&tokenizer, path);

    active_injection = INJECT_NONE;
    expect(injection_hits == 1 && mutation_succeeded,
           "the source file changes after its bounded snapshot");
    expect(model != NULL && tokenizer != NULL,
           "checkpoint load parses the immutable snapshot");
    expect(!checkpoint_loads(path),
           "the corrupted on-disk replacement does not load later");

    if (model != NULL)
        model_free(model);
    if (tokenizer != NULL)
        tokenizer_free(tokenizer);
    unlink(path);
}

static void check_fmemopen_failure(Fixture fixture, const char *path)
{
    expect(model_save_durable(fixture.model, fixture.tokenizer, path)
               == MODEL_SAVE_DURABLE,
           "memory-stream failure fixture saves");

    start_injection(INJECT_FMEMOPEN);
    Tokenizer *tokenizer = fixture.tokenizer;
    Model *model = model_load(&tokenizer, path);
    active_injection = INJECT_NONE;

    expect(injection_hits == 1,
           "memory-stream construction failure point is reached");
    expect(model == NULL && tokenizer == NULL,
           "memory-stream failure publishes no load result");
    unlink(path);
}

static void check_registry_preflight(Fixture fixture, const char *path)
{
    expect(replace_with_bytes(path, OLD_BYTES, sizeof OLD_BYTES) == 0,
           "registry preflight destination is ready");

    int saved_count = fixture.model->param_count;

    fixture.model->param_count--;
    ModelSaveResult result =
        model_save_durable(fixture.model, fixture.tokenizer, path);
    fixture.model->param_count = saved_count;

    expect(result == MODEL_SAVE_NOT_COMMITTED,
           "save rejects a registry scalar-count mismatch");
    expect(file_equals(path, OLD_BYTES, sizeof OLD_BYTES),
           "registry rejection leaves the destination untouched");
    unlink(path);
}

static void check_temporary_identity(Fixture fixture, const char *directory,
                                     const char *path)
{
    expect(replace_with_bytes(path, OLD_BYTES, sizeof OLD_BYTES) == 0,
           "temporary-identity destination is ready");

    mutation_directory = directory;
    mutation_succeeded = 0;
    swapped_temp_path[0] = '\0';
    displaced_temp_path[0] = '\0';
    start_injection(MUTATE_TEMP_AFTER_CLOSE);
    ModelSaveResult result =
        model_save_durable(fixture.model, fixture.tokenizer, path);
    active_injection = INJECT_NONE;

    expect(injection_hits == 1 && mutation_succeeded,
           "the closed temporary entry is replaced before commit");
    expect(result == MODEL_SAVE_NOT_COMMITTED,
           "a replaced temporary entry is not committed");
    expect(file_equals(path, OLD_BYTES, sizeof OLD_BYTES),
           "temporary replacement leaves the destination untouched");
    expect(file_equals(swapped_temp_path, OLD_BYTES, sizeof OLD_BYTES),
           "cleanup does not unlink the foreign temporary replacement");
    expect(checkpoint_loads(displaced_temp_path),
           "the displaced owned temporary file contains the checkpoint");

    unlink(swapped_temp_path);
    unlink(displaced_temp_path);
    unlink(path);
}

static uint64_t fnv1a64(const unsigned char *bytes, size_t size)
{
    uint64_t hash = UINT64_C(14695981039346656037);

    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int golden_native_format_matches(void)
{
    uint32_t integer = UINT32_C(1);
    uint32_t float_bits = 0;
    float one = 1.0f;

    if (sizeof one != sizeof float_bits)
        return 0;
    memcpy(&float_bits, &one, sizeof float_bits);
    return *(const unsigned char *)&integer == 1
        && float_bits == UINT32_C(0x3f800000);
}

static void check_writer_golden(const char *path)
{
    /*
     * TAGC v1 is native-format.  This historical byte witness applies
     * only to its recorded little-endian IEEE binary32 platform.
     */
    if (!golden_native_format_matches())
        return;

    static const char TEXT[] = "x";
    Tokenizer *tokenizer = tokenizer_new(TEXT, sizeof TEXT - 1);
    ModelConfig config = {
        .vocab_size  = 1,
        .block_size  = 1,
        .d_model     = 1,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    Model *model = model_new(config, GOLDEN_MODEL_SEED);
    ModelParams params = model_params(model);
    size_t scalar = 0;

    for (int p = 0; p < params.count; p++) {
        Mat values = param_values(params.params[p]);

        for (size_t i = 0; i < mat_size(values); i++) {
            values.vals[i] =
                ((int)(scalar % GOLDEN_VALUE_PERIOD) - GOLDEN_VALUE_CENTER)
                * GOLDEN_VALUE_SCALE;
            scalar++;
        }
    }
    expect(scalar == GOLDEN_PARAMETER_SCALARS,
           "golden fixture has 20 parameter scalars");
    expect(model_save_durable(model, tokenizer, path)
               == MODEL_SAVE_DURABLE,
           "golden checkpoint saves durably");

    char *bytes = NULL;
    size_t size = 0;

    expect(file_slurp_bounded(path, GOLDEN_CHECKPOINT_BYTES,
                              &bytes, &size) == FILE_SLURP_OK,
           "golden checkpoint fits its exact historical size");
    if (bytes != NULL) {
        uint64_t hash = fnv1a64((const unsigned char *)bytes, size);

        expect(size == GOLDEN_CHECKPOINT_BYTES
                   && hash == GOLDEN_CHECKPOINT_FNV1A,
               "golden checkpoint keeps the TAGC v1 writer bytes");
        free(bytes);
    }

    unlink(path);
    model_free(model);
    tokenizer_free(tokenizer);
}

static int checkpoint_failure_paths(const char *directory,
                                    CheckpointFailurePaths *paths)
{
    return join_path(paths->metadata, sizeof paths->metadata,
                     directory, "metadata.bin") != 0
        || join_path(paths->snapshot, sizeof paths->snapshot,
                     directory, "snapshot.bin") != 0
        || join_path(paths->registry, sizeof paths->registry,
                     directory, "registry.bin") != 0
        || join_path(paths->identity, sizeof paths->identity,
                     directory, "identity.bin") != 0
        || join_path(paths->golden, sizeof paths->golden,
                     directory, "golden.bin") != 0
        || join_path(paths->fmemopen, sizeof paths->fmemopen,
                     directory, "fmemopen.bin") != 0
        ? -1 : 0;
}

int main(void)
{
    char directory[] = "/tmp/tiny-agenc-checkpoint-failures-XXXXXX";

    if (mkdtemp(directory) == NULL) {
        perror("mkdtemp");
        return EXIT_FAILURE;
    }

    CheckpointFailurePaths paths;

    if (checkpoint_failure_paths(directory, &paths) != 0) {
        fprintf(stderr, "checkpoint failure: temporary path is too long\n");
        rmdir(directory);
        return EXIT_FAILURE;
    }

    Fixture fixture = fixture_new();

    check_metadata_round_trip(fixture, paths.metadata);
    unlink(paths.metadata);
    check_rejected_destinations(fixture, directory);
    check_injected_save_failures(fixture, directory);
    check_immutable_load_snapshot(fixture, paths.snapshot);
    check_fmemopen_failure(fixture, paths.fmemopen);
    check_registry_preflight(fixture, paths.registry);
    check_temporary_identity(fixture, directory, paths.identity);
    check_writer_golden(paths.golden);

    fixture_free(fixture);
    expect(directory_entry_count(directory) == 0,
           "all checkpoint test files are cleaned up");
    if (rmdir(directory) != 0) {
        perror("rmdir");
        failures++;
    }

    if (failures != 0) {
        fprintf(stderr, "checkpoint failures: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("checkpoint failures: all %d checks passed\n", checks);
    return EXIT_SUCCESS;
}
