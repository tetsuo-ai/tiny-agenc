#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <acl/libacl.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/acl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>

#include "model_internal.h"
#include "util.h"

static const uint32_t CHECKPOINT_MAGIC   = 0x43474154u; /* "TAGC" */
static const int32_t  CHECKPOINT_VERSION = 1;

enum {
    CHECKPOINT_HEADER_I32S = 8,
    CHECKPOINT_TOKENIZER_SIZE_I32S = 1,
    CHECKPOINT_CHECKSUM_I32S = 1,
    CHECKSUM_ENVELOPE_I32S = 3,   /* magic, version, and checksum */
    CRC32_BITS_PER_BYTE = 8,
    CRC32_BUFFER_BYTES = 8192,
    XATTR_READ_ATTEMPTS = 3,
    FILE_PERMISSION_MASK = 07777,
    PRIVATE_FILE_MODE = 0600,
    TEMP_RANDOM_BYTES = 16,
    TEMP_CREATE_ATTEMPTS = 16,
    HEX_DIGITS_PER_BYTE = 2,
    HEX_NIBBLE_BITS = 4,
    HEX_NIBBLE_MASK = (1 << HEX_NIBBLE_BITS) - 1,
    LOWEST_DUPLICATE_DESCRIPTOR = 0,
};

static const uint32_t CRC32_POLYNOMIAL = 0xEDB88320u;
static const uint32_t CRC32_INITIAL = 0xFFFFFFFFu;

typedef enum {
    XATTR_REGULAR_STAGE,
    XATTR_CAPABILITY_STAGE,
    XATTR_SELINUX_STAGE,
    XATTR_STAGE_COUNT,
} XattrApplyStage;

typedef struct {
    ModelMemory memory;
    size_t      parameter_floats;
    size_t      parameter_bytes;
    size_t      file_bytes;
} CheckpointLayout;

typedef struct {
    char   *name;
    void   *value;
    size_t  size;
} SavedXattr;

typedef struct {
    uid_t       uid;
    gid_t       gid;
    mode_t      mode;
    acl_t       access_acl;
    SavedXattr *xattrs;
    size_t      xattr_count;
} CheckpointMetadata;

typedef struct {
    char       *directory_name;
    char       *name;
    int         directory;
    int         target;
    int         target_exists;
    struct stat target_stat;
} SavePath;

typedef struct {
    SavePath           path;
    CheckpointMetadata metadata;
    char                temp_name[
        sizeof ".tiny-agenc-" + HEX_DIGITS_PER_BYTE * TEMP_RANDOM_BYTES];
    int                 temp_descriptor;
    int                 temp_identity_descriptor;
    FILE               *stream;
    struct stat         temp_stat;
    int                 has_temp_identity;
    int                 renamed;
} SaveTransaction;

typedef struct {
    char      *snapshot;
    size_t     snapshot_size;
    FILE      *stream;
    Tokenizer *tokenizer;
    Model     *model;
} LoadCandidate;

static size_t checkpoint_fixed_bytes(void);
static int checkpoint_layout(ModelConfig cfg, CheckpointLayout *layout);
static uint32_t crc32_update(uint32_t crc, const unsigned char *bytes,
                             size_t count);
static int crc32_prefix(FILE *stream, off_t length, uint32_t *result);
static int append_checksum(FILE *stream);
static int snapshot_checksum_matches(const char *snapshot, size_t size);
static int write_checkpoint_header(FILE *stream, const Model *m);
static int write_checkpoint_parameters(FILE *stream, const Model *m);
static int write_checkpoint(FILE *stream, const Model *m,
                            const Tokenizer *tk);
static char *copy_string_range(const char *text, size_t length);
static void save_path_init(SavePath *path);
static int split_save_path(SavePath *save_path, const char *path);
static int stat_destination(SavePath *path);
static int open_existing_destination(SavePath *path);
static int prepare_save_path(SavePath *save_path, const char *path);
static int destination_identity_matches(const SavePath *path);
static void save_path_free(SavePath *path);
static int xattr_is_posix_access_acl(const char *name);
static int xattr_is_unsupported(const char *name);
static int read_xattr_names(int descriptor, char **names, size_t *size);
static int count_managed_xattrs(const char *names, size_t size,
                                size_t *count);
static int read_xattr_value(int descriptor, const char *name,
                            void **value, size_t *size);
static int capture_one_xattr(int descriptor, const char *name,
                             SavedXattr *saved);
static void free_saved_xattrs(CheckpointMetadata *metadata);
static int capture_named_xattrs(int descriptor, const char *names,
                                size_t names_size,
                                CheckpointMetadata *metadata);
static int capture_xattrs(int descriptor, CheckpointMetadata *metadata);
static int capture_metadata(int descriptor, CheckpointMetadata *metadata);
static const SavedXattr *find_saved_xattr(
    const CheckpointMetadata *metadata, const char *name);
static XattrApplyStage xattr_apply_stage(const char *name);
static int apply_saved_xattrs(int descriptor,
                              const CheckpointMetadata *metadata);
static int remove_unmatched_xattrs(
    int descriptor, const CheckpointMetadata *metadata);
static int saved_xattr_matches(int descriptor, const SavedXattr *saved);
static int xattrs_match(int descriptor,
                        const CheckpointMetadata *metadata);
static int metadata_matches(int descriptor,
                            const CheckpointMetadata *metadata);
static int apply_metadata(int descriptor,
                          const CheckpointMetadata *metadata);
static void metadata_free(CheckpointMetadata *metadata);
static int random_bytes(unsigned char *bytes, size_t count);
static void format_temp_name(
    char *name, const unsigned char random[TEMP_RANDOM_BYTES]);
static int record_temporary_identity(SaveTransaction *transaction);
static void discard_temporary_file(SaveTransaction *transaction,
                                   int saved_error);
static int create_temporary_file(SaveTransaction *transaction);
static int path_matches_temporary_file(const SaveTransaction *transaction,
                                       const char *name);
static void transaction_init(SaveTransaction *transaction);
static int transaction_close_stream(SaveTransaction *transaction);
static int prepare_transaction(SaveTransaction *transaction,
                               const char *path);
static int prepare_temporary_checkpoint(
    SaveTransaction *transaction, const Model *m, const Tokenizer *tk);
static int close_original_destination(SaveTransaction *transaction);
static int commit_temporary_checkpoint(SaveTransaction *transaction);
static ModelSaveResult confirm_directory_update(
    SaveTransaction *transaction);
static void remove_uncommitted_temporary_file(
    SaveTransaction *transaction);
static void transaction_cleanup(SaveTransaction *transaction);
static int read_dimension(FILE *stream, int *value);
static int read_config(FILE *stream, ModelConfig *cfg);
static void load_candidate_init(LoadCandidate *candidate);
static void load_candidate_free(LoadCandidate *candidate);
static int load_snapshot(LoadCandidate *candidate, const char *path);
static int snapshot_fits_resident_limit(
    const LoadCandidate *candidate, const CheckpointLayout *layout);
static int load_header_and_tokenizer(
    LoadCandidate *candidate, CheckpointLayout *layout);
static int load_checkpoint_parameters(LoadCandidate *candidate);
static int consume_checkpoint_trailer(LoadCandidate *candidate);
static Model *publish_load_candidate(LoadCandidate *candidate,
                                     Tokenizer **tokenizer);

static size_t checkpoint_fixed_bytes(void)
{
    return (CHECKPOINT_HEADER_I32S + CHECKPOINT_TOKENIZER_SIZE_I32S
            + CHECKPOINT_CHECKSUM_I32S) * sizeof(int32_t);
}

static int checkpoint_layout(ModelConfig cfg, CheckpointLayout *layout)
{
    CheckpointLayout candidate = {0};
    size_t fixed_bytes = checkpoint_fixed_bytes();

    candidate.parameter_floats = model_parameter_float_count(cfg);
    if (!model_memory_requirements(cfg, &candidate.memory)
        || candidate.parameter_floats > SIZE_MAX / sizeof(float))
        return 0;

    candidate.parameter_bytes =
        candidate.parameter_floats * sizeof(float);
    if (fixed_bytes > MODEL_MAX_CHECKPOINT_FILE_BYTES
        || (size_t)cfg.vocab_size
           > MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
        || candidate.parameter_bytes
           > MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
            - (size_t)cfg.vocab_size)
        return 0;

    candidate.file_bytes =
        fixed_bytes + (size_t)cfg.vocab_size + candidate.parameter_bytes;
    if (candidate.memory.total_bytes
        > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES)
        return 0;
    *layout = candidate;
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

static int crc32_prefix(FILE *stream, off_t length, uint32_t *result)
{
    unsigned char buffer[CRC32_BUFFER_BYTES];
    uint32_t      crc = CRC32_INITIAL;

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

    if (payload_end < 0
        || crc32_prefix(stream, payload_end, &checksum) != 0
        || fseeko(stream, payload_end, SEEK_SET) != 0
        || write_i32(stream, (int32_t)checksum) != 0)
        return -1;
    return fflush(stream);
}

static int snapshot_checksum_matches(const char *snapshot, size_t size)
{
    if (size < CHECKSUM_ENVELOPE_I32S * sizeof(int32_t))
        return 0;

    size_t checksum_at = size - sizeof(int32_t);
    int32_t stored;
    uint32_t computed =
        ~crc32_update(CRC32_INITIAL,
                      (const unsigned char *)snapshot, checksum_at);

    memcpy(&stored, snapshot + checksum_at, sizeof stored);
    return computed == (uint32_t)stored;
}

static int write_checkpoint_header(FILE *stream, const Model *m)
{
    return write_i32(stream, (int32_t)CHECKPOINT_MAGIC) != 0
        || write_i32(stream, CHECKPOINT_VERSION) != 0
        || write_i32(stream, m->cfg.vocab_size) != 0
        || write_i32(stream, m->cfg.block_size) != 0
        || write_i32(stream, m->cfg.d_model) != 0
        || write_i32(stream, m->cfg.head_count) != 0
        || write_i32(stream, m->cfg.layer_count) != 0
        || write_i32(stream, m->cfg.batch_size) != 0;
}

static int write_checkpoint_parameters(FILE *stream, const Model *m)
{
    for (int i = 0; i < m->param_count; i++) {
        if (param_write(m->params[i], stream) != 0)
            return -1;
    }
    return 0;
}

static int write_checkpoint(FILE *stream, const Model *m,
                            const Tokenizer *tk)
{
    if (write_checkpoint_header(stream, m) != 0
        || tokenizer_write(tk, stream) != 0
        || write_checkpoint_parameters(stream, m) != 0)
        return -1;
    return append_checksum(stream);
}

static char *copy_string_range(const char *text, size_t length)
{
    char *copy = emalloc(length + 1);

    memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

static void save_path_init(SavePath *path)
{
    memset(path, 0, sizeof *path);
    path->directory = -1;
    path->target = -1;
}

static int split_save_path(SavePath *save_path, const char *path)
{
    size_t length = strlen(path);
    const char *slash = strrchr(path, '/');
    const char *name = slash == NULL ? path : slash + 1;

    if (length == 0 || name[0] == '\0'
        || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
        return -1;

    if (slash == NULL)
        save_path->directory_name =
            copy_string_range(".", sizeof "." - 1);
    else if (slash == path)
        save_path->directory_name =
            copy_string_range("/", sizeof "/" - 1);
    else
        save_path->directory_name =
            copy_string_range(path, (size_t)(slash - path));
    save_path->name = copy_string_range(name, strlen(name));
    return 0;
}

static int stat_destination(SavePath *path)
{
    if (fstatat(path->directory, path->name, &path->target_stat,
                AT_SYMLINK_NOFOLLOW) == 0) {
        path->target_exists = 1;
        return S_ISREG(path->target_stat.st_mode) ? 0 : -1;
    }
    if (errno != ENOENT)
        return -1;
    path->target_exists = 0;
    return 0;
}

static int open_existing_destination(SavePath *path)
{
    if (!path->target_exists)
        return 0;

    path->target =
        openat(path->directory, path->name,
               O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
    if (path->target < 0)
        return -1;

    struct stat opened;

    if (fstat(path->target, &opened) != 0 || !S_ISREG(opened.st_mode)
        || opened.st_dev != path->target_stat.st_dev
        || opened.st_ino != path->target_stat.st_ino)
        return -1;
    return 0;
}

static int prepare_save_path(SavePath *save_path, const char *path)
{
    if (path == NULL || split_save_path(save_path, path) != 0)
        return -1;

    save_path->directory =
        open(save_path->directory_name,
             O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (save_path->directory < 0 || stat_destination(save_path) != 0)
        return -1;
    return open_existing_destination(save_path);
}

static int destination_identity_matches(const SavePath *path)
{
    struct stat current;

    if (!path->target_exists)
        return 1;
    if (fstatat(path->directory, path->name, &current,
                AT_SYMLINK_NOFOLLOW) != 0)
        return 0;
    return S_ISREG(current.st_mode)
        && current.st_dev == path->target_stat.st_dev
        && current.st_ino == path->target_stat.st_ino;
}

static void save_path_free(SavePath *path)
{
    if (path->target >= 0)
        close(path->target);
    if (path->directory >= 0)
        close(path->directory);
    free(path->name);
    free(path->directory_name);
    save_path_init(path);
}

static int xattr_is_posix_access_acl(const char *name)
{
    return strcmp(name, "system.posix_acl_access") == 0;
}

static int xattr_is_unsupported(const char *name)
{
    return strcmp(name, "system.posix_acl_default") == 0
        || strcmp(name, "system.nfs4_acl") == 0
        || strcmp(name, "system.nfs4acl") == 0
        || strcmp(name, "trusted.nfs4_acl") == 0
        || strcmp(name, "security.ima") == 0
        || strcmp(name, "security.evm") == 0;
}

static int read_xattr_names(int descriptor, char **names, size_t *size)
{
    *names = NULL;
    *size = 0;

    for (int attempt = 0; attempt < XATTR_READ_ATTEMPTS; attempt++) {
        ssize_t wanted = flistxattr(descriptor, NULL, 0);

        if (wanted < 0)
            return -1;
        if (wanted == 0)
            return 0;

        char *buffer = emalloc((size_t)wanted);
        ssize_t received =
            flistxattr(descriptor, buffer, (size_t)wanted);

        if (received >= 0) {
            *names = buffer;
            *size = (size_t)received;
            return 0;
        }
        free(buffer);
        if (errno != ERANGE)
            return -1;
    }
    return -1;
}

static int count_managed_xattrs(const char *names, size_t size,
                                size_t *count)
{
    size_t position = 0;
    size_t found = 0;

    while (position < size) {
        size_t remaining = size - position;
        size_t length = strnlen(names + position, remaining);

        if (length == 0 || length == remaining)
            return -1;
        if (xattr_is_unsupported(names + position))
            return -1;
        if (!xattr_is_posix_access_acl(names + position))
            found++;
        position += length + 1;
    }
    *count = found;
    return 0;
}

static int read_xattr_value(int descriptor, const char *name,
                            void **value, size_t *size)
{
    for (int attempt = 0; attempt < XATTR_READ_ATTEMPTS; attempt++) {
        ssize_t wanted = fgetxattr(descriptor, name, NULL, 0);

        if (wanted < 0)
            return -1;

        void *buffer = emalloc(wanted == 0 ? 1 : (size_t)wanted);
        ssize_t received =
            fgetxattr(descriptor, name, buffer, (size_t)wanted);

        if (received == wanted) {
            *value = buffer;
            *size = (size_t)wanted;
            return 0;
        }
        free(buffer);
        if (received >= 0 || errno != ERANGE)
            return -1;
    }
    return -1;
}

static int capture_one_xattr(int descriptor, const char *name,
                             SavedXattr *saved)
{
    saved->name = copy_string_range(name, strlen(name));
    if (read_xattr_value(descriptor, name, &saved->value,
                         &saved->size) != 0) {
        free(saved->name);
        saved->name = NULL;
        return -1;
    }
    return 0;
}

static void free_saved_xattrs(CheckpointMetadata *metadata)
{
    for (size_t i = 0; i < metadata->xattr_count; i++) {
        free(metadata->xattrs[i].name);
        free(metadata->xattrs[i].value);
    }
    free(metadata->xattrs);
    metadata->xattrs = NULL;
    metadata->xattr_count = 0;
}

static int capture_named_xattrs(int descriptor, const char *names,
                                size_t names_size,
                                CheckpointMetadata *metadata)
{
    size_t position = 0;
    size_t saved = 0;

    while (position < names_size) {
        const char *name = names + position;
        size_t length = strlen(name);

        position += length + 1;
        if (xattr_is_posix_access_acl(name))
            continue;
        if (capture_one_xattr(descriptor, name,
                              &metadata->xattrs[saved]) != 0) {
            metadata->xattr_count = saved;
            free_saved_xattrs(metadata);
            return -1;
        }
        saved++;
    }
    metadata->xattr_count = saved;
    return 0;
}

static int capture_xattrs(int descriptor, CheckpointMetadata *metadata)
{
    char *names;
    size_t names_size;
    size_t count;

    if (read_xattr_names(descriptor, &names, &names_size) != 0)
        return -1;
    if (count_managed_xattrs(names, names_size, &count) != 0) {
        free(names);
        return -1;
    }
    if (count > 0)
        metadata->xattrs = ecalloc(count, sizeof *metadata->xattrs);

    int result =
        capture_named_xattrs(descriptor, names, names_size, metadata);

    free(names);
    return result;
}

static int capture_metadata(int descriptor,
                            CheckpointMetadata *metadata)
{
    struct stat status;

    if (fstat(descriptor, &status) != 0)
        return -1;
    metadata->uid = status.st_uid;
    metadata->gid = status.st_gid;
    metadata->mode = status.st_mode & FILE_PERMISSION_MASK;
    metadata->access_acl = acl_get_fd(descriptor);
    if (metadata->access_acl == NULL)
        return -1;
    return capture_xattrs(descriptor, metadata);
}

static const SavedXattr *find_saved_xattr(
    const CheckpointMetadata *metadata, const char *name)
{
    for (size_t i = 0; i < metadata->xattr_count; i++) {
        if (strcmp(metadata->xattrs[i].name, name) == 0)
            return &metadata->xattrs[i];
    }
    return NULL;
}

static XattrApplyStage xattr_apply_stage(const char *name)
{
    if (strcmp(name, "security.capability") == 0)
        return XATTR_CAPABILITY_STAGE;
    if (strcmp(name, "security.selinux") == 0)
        return XATTR_SELINUX_STAGE;
    return XATTR_REGULAR_STAGE;
}

static int apply_saved_xattrs(int descriptor,
                              const CheckpointMetadata *metadata)
{
    for (XattrApplyStage stage = XATTR_REGULAR_STAGE;
         stage < XATTR_STAGE_COUNT; stage++) {
        for (size_t i = 0; i < metadata->xattr_count; i++) {
            const SavedXattr *xattr = &metadata->xattrs[i];

            if (xattr_apply_stage(xattr->name) == stage
                && fsetxattr(descriptor, xattr->name, xattr->value,
                             xattr->size, 0) != 0)
                return -1;
        }
    }
    return 0;
}

static int remove_unmatched_xattrs(
    int descriptor, const CheckpointMetadata *metadata)
{
    char *names;
    size_t names_size;

    if (read_xattr_names(descriptor, &names, &names_size) != 0)
        return -1;

    size_t position = 0;

    while (position < names_size) {
        const char *name = names + position;
        size_t length = strlen(name);

        if (xattr_is_unsupported(name)) {
            free(names);
            return -1;
        }
        if (!xattr_is_posix_access_acl(name)
            && find_saved_xattr(metadata, name) == NULL
            && fremovexattr(descriptor, name) != 0) {
            free(names);
            return -1;
        }
        position += length + 1;
    }
    free(names);
    return 0;
}

static int saved_xattr_matches(int descriptor,
                               const SavedXattr *saved)
{
    void *value;
    size_t size;

    if (read_xattr_value(descriptor, saved->name, &value, &size) != 0)
        return 0;
    int matches =
        size == saved->size && memcmp(value, saved->value, size) == 0;

    free(value);
    return matches;
}

static int xattrs_match(int descriptor,
                        const CheckpointMetadata *metadata)
{
    char *names;
    size_t names_size;
    size_t count;

    if (read_xattr_names(descriptor, &names, &names_size) != 0)
        return 0;
    if (count_managed_xattrs(names, names_size, &count) != 0
        || count != metadata->xattr_count) {
        free(names);
        return 0;
    }
    free(names);

    for (size_t i = 0; i < metadata->xattr_count; i++) {
        if (!saved_xattr_matches(descriptor, &metadata->xattrs[i]))
            return 0;
    }
    return 1;
}

static int metadata_matches(int descriptor,
                            const CheckpointMetadata *metadata)
{
    struct stat status;
    acl_t access_acl;

    if (fstat(descriptor, &status) != 0
        || status.st_uid != metadata->uid
        || status.st_gid != metadata->gid
        || (status.st_mode & FILE_PERMISSION_MASK) != metadata->mode)
        return 0;

    access_acl = acl_get_fd(descriptor);
    if (access_acl == NULL)
        return 0;
    int acl_matches = acl_cmp(access_acl, metadata->access_acl) == 0;

    acl_free(access_acl);
    return acl_matches && xattrs_match(descriptor, metadata);
}

static int apply_metadata(int descriptor,
                          const CheckpointMetadata *metadata)
{
    if (fchown(descriptor, metadata->uid, metadata->gid) != 0
        || acl_set_fd(descriptor, metadata->access_acl) != 0
        || fchmod(descriptor, metadata->mode) != 0
        || apply_saved_xattrs(descriptor, metadata) != 0
        || remove_unmatched_xattrs(descriptor, metadata) != 0)
        return -1;
    return metadata_matches(descriptor, metadata) ? 0 : -1;
}

static void metadata_free(CheckpointMetadata *metadata)
{
    if (metadata->access_acl != NULL)
        acl_free(metadata->access_acl);
    metadata->access_acl = NULL;
    free_saved_xattrs(metadata);
}

static int random_bytes(unsigned char *bytes, size_t count)
{
    size_t filled = 0;

    while (filled < count) {
        ssize_t received = getrandom(bytes + filled, count - filled, 0);

        if (received > 0) {
            filled += (size_t)received;
            continue;
        }
        if (received < 0 && errno == EINTR)
            continue;
        return -1;
    }
    return 0;
}

static void format_temp_name(char *name,
                             const unsigned char random[TEMP_RANDOM_BYTES])
{
    static const char HEX[] = "0123456789abcdef";
    static const char PREFIX[] = ".tiny-agenc-";
    size_t prefix_length = sizeof PREFIX - 1;

    memcpy(name, PREFIX, prefix_length);
    for (size_t i = 0; i < TEMP_RANDOM_BYTES; i++) {
        name[prefix_length + HEX_DIGITS_PER_BYTE * i] =
            HEX[random[i] >> HEX_NIBBLE_BITS];
        name[prefix_length + HEX_DIGITS_PER_BYTE * i + 1] =
            HEX[random[i] & HEX_NIBBLE_MASK];
    }
    name[prefix_length + HEX_DIGITS_PER_BYTE * TEMP_RANDOM_BYTES] = '\0';
}

static int record_temporary_identity(SaveTransaction *transaction)
{
    transaction->temp_identity_descriptor =
        fcntl(transaction->temp_descriptor, F_DUPFD_CLOEXEC,
              LOWEST_DUPLICATE_DESCRIPTOR);
    if (transaction->temp_identity_descriptor < 0)
        return -1;
    if (fstat(transaction->temp_identity_descriptor,
              &transaction->temp_stat) != 0)
        return -1;
    if (!S_ISREG(transaction->temp_stat.st_mode)) {
        errno = EINVAL;
        return -1;
    }
    transaction->has_temp_identity = 1;
    return 0;
}

static void discard_temporary_file(SaveTransaction *transaction,
                                   int saved_error)
{
    if (transaction->temp_identity_descriptor >= 0)
        close(transaction->temp_identity_descriptor);
    transaction->temp_identity_descriptor = -1;

    unlinkat(transaction->path.directory, transaction->temp_name, 0);
    close(transaction->temp_descriptor);
    transaction->temp_descriptor = -1;
    transaction->temp_name[0] = '\0';
    errno = saved_error;
}

static int create_temporary_file(SaveTransaction *transaction)
{
    unsigned char random[TEMP_RANDOM_BYTES];

    for (int attempt = 0; attempt < TEMP_CREATE_ATTEMPTS; attempt++) {
        if (random_bytes(random, sizeof random) != 0)
            return -1;
        format_temp_name(transaction->temp_name, random);
        transaction->temp_descriptor =
            openat(transaction->path.directory, transaction->temp_name,
                   O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
                   PRIVATE_FILE_MODE);
        if (transaction->temp_descriptor < 0) {
            if (errno == EEXIST)
                continue;
            return -1;
        }
        if (record_temporary_identity(transaction) == 0)
            return 0;

        int saved_error = errno;

        discard_temporary_file(transaction, saved_error);
        return -1;
    }
    return -1;
}

static int path_matches_temporary_file(const SaveTransaction *transaction,
                                       const char *name)
{
    struct stat status;

    /*
     * The duplicate descriptor keeps the owned inode alive, preventing
     * inode-number reuse while the name is checked.  Linux has no
     * rename- or unlink-by-inode primitive, so callers must still exclude
     * same-authority mutation of this unguessable temporary name.
     */
    if (!transaction->has_temp_identity
        || fstatat(transaction->path.directory, name, &status,
                   AT_SYMLINK_NOFOLLOW) != 0)
        return 0;
    return S_ISREG(status.st_mode)
        && status.st_dev == transaction->temp_stat.st_dev
        && status.st_ino == transaction->temp_stat.st_ino;
}

static void transaction_init(SaveTransaction *transaction)
{
    memset(transaction, 0, sizeof *transaction);
    save_path_init(&transaction->path);
    transaction->temp_descriptor = -1;
    transaction->temp_identity_descriptor = -1;
}

static int transaction_close_stream(SaveTransaction *transaction)
{
    FILE *stream = transaction->stream;

    transaction->stream = NULL;
    return fclose(stream);
}

static int prepare_transaction(SaveTransaction *transaction,
                               const char *path)
{
    if (prepare_save_path(&transaction->path, path) != 0)
        return -1;
    if (transaction->path.target_exists
        && capture_metadata(transaction->path.target,
                            &transaction->metadata) != 0)
        return -1;

    /*
     * Probe directory-sync support before creating or replacing anything.
     * The final sync after rename is still the durability boundary.
     */
    if (fsync(transaction->path.directory) != 0
        || create_temporary_file(transaction) != 0)
        return -1;

    transaction->stream =
        fdopen(transaction->temp_descriptor, "w+b");
    if (transaction->stream == NULL)
        return -1;
    transaction->temp_descriptor = -1;
    return 0;
}

static int prepare_temporary_checkpoint(
    SaveTransaction *transaction, const Model *m, const Tokenizer *tk)
{
    if (write_checkpoint(transaction->stream, m, tk) != 0)
        return -1;

    int descriptor = fileno(transaction->stream);

    if (descriptor < 0)
        return -1;
    if ((transaction->path.target_exists
         && apply_metadata(descriptor, &transaction->metadata) != 0)
        || (!transaction->path.target_exists
            && fchmod(descriptor, PRIVATE_FILE_MODE) != 0))
        return -1;
    if (fsync(descriptor) != 0)
        return -1;
    return transaction_close_stream(transaction);
}

static int close_original_destination(SaveTransaction *transaction)
{
    if (!transaction->path.target_exists)
        return 0;
    if (!metadata_matches(transaction->path.target,
                          &transaction->metadata)
        || !destination_identity_matches(&transaction->path))
        return -1;

    int descriptor = transaction->path.target;

    transaction->path.target = -1;
    return close(descriptor);
}

static int commit_temporary_checkpoint(SaveTransaction *transaction)
{
    int renamed;

    if (close_original_destination(transaction) != 0
        || !path_matches_temporary_file(transaction,
                                        transaction->temp_name))
        return -1;
    if (transaction->path.target_exists) {
        renamed =
            renameat(transaction->path.directory, transaction->temp_name,
                     transaction->path.directory,
                     transaction->path.name);
    } else {
        renamed =
            renameat2(transaction->path.directory, transaction->temp_name,
                      transaction->path.directory,
                      transaction->path.name, RENAME_NOREPLACE);
    }
    if (renamed != 0)
        return -1;
    transaction->renamed = 1;
    transaction->temp_name[0] = '\0';
    return 0;
}

static ModelSaveResult confirm_directory_update(
    SaveTransaction *transaction)
{
    int sync_failed = fsync(transaction->path.directory) != 0;
    int descriptor = transaction->path.directory;

    transaction->path.directory = -1;
    if (close(descriptor) != 0)
        sync_failed = 1;
    return sync_failed
        ? MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED
        : MODEL_SAVE_DURABLE;
}

static void remove_uncommitted_temporary_file(
    SaveTransaction *transaction)
{
    if (transaction->renamed || transaction->temp_name[0] == '\0'
        || transaction->path.directory < 0)
        return;
    if (!path_matches_temporary_file(transaction, transaction->temp_name))
        return;
    unlinkat(transaction->path.directory, transaction->temp_name, 0);
}

static void transaction_cleanup(SaveTransaction *transaction)
{
    if (transaction->stream != NULL)
        fclose(transaction->stream);
    else if (transaction->temp_descriptor >= 0)
        close(transaction->temp_descriptor);
    remove_uncommitted_temporary_file(transaction);
    if (transaction->temp_identity_descriptor >= 0)
        close(transaction->temp_identity_descriptor);
    metadata_free(&transaction->metadata);
    save_path_free(&transaction->path);
}

ModelSaveResult model_save_durable(const Model *m, const Tokenizer *tk,
                                   const char *path)
{
    CheckpointLayout layout;

    if (m == NULL || tk == NULL || path == NULL
        || !checkpoint_layout(m->cfg, &layout)
        || model_parameter_count(m) != layout.parameter_floats
        || tokenizer_vocab_size(tk) != m->cfg.vocab_size)
        return MODEL_SAVE_NOT_COMMITTED;

    SaveTransaction transaction;
    ModelSaveResult result = MODEL_SAVE_NOT_COMMITTED;

    transaction_init(&transaction);
    if (prepare_transaction(&transaction, path) != 0
        || prepare_temporary_checkpoint(&transaction, m, tk) != 0
        || commit_temporary_checkpoint(&transaction) != 0)
        goto done;

    result = confirm_directory_update(&transaction);

done:
    transaction_cleanup(&transaction);
    return result;
}

int model_save(const Model *m, const Tokenizer *tk, const char *path)
{
    return model_save_durable(m, tk, path) == MODEL_SAVE_DURABLE
        ? 0 : -1;
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

static void load_candidate_init(LoadCandidate *candidate)
{
    memset(candidate, 0, sizeof *candidate);
}

static void load_candidate_free(LoadCandidate *candidate)
{
    if (candidate->stream != NULL)
        fclose(candidate->stream);
    if (candidate->model != NULL)
        model_free(candidate->model);
    if (candidate->tokenizer != NULL)
        tokenizer_free(candidate->tokenizer);
    free(candidate->snapshot);
    load_candidate_init(candidate);
}

static int load_snapshot(LoadCandidate *candidate, const char *path)
{
    if (path == NULL
        || file_slurp_bounded(path, MODEL_MAX_CHECKPOINT_FILE_BYTES,
                              &candidate->snapshot,
                              &candidate->snapshot_size)
           != FILE_SLURP_OK
        || !snapshot_checksum_matches(candidate->snapshot,
                                      candidate->snapshot_size))
        return -1;

    candidate->stream =
        fmemopen(candidate->snapshot, candidate->snapshot_size, "rb");
    return candidate->stream == NULL ? -1 : 0;
}

static int snapshot_fits_resident_limit(
    const LoadCandidate *candidate, const CheckpointLayout *layout)
{
    if (candidate->snapshot_size == SIZE_MAX)
        return 0;

    size_t allocated_snapshot = candidate->snapshot_size + 1;

    return allocated_snapshot <= MODEL_MAX_CHECKPOINT_RESIDENT_BYTES
        && layout->memory.total_bytes
           <= MODEL_MAX_CHECKPOINT_RESIDENT_BYTES - allocated_snapshot;
}

static int load_header_and_tokenizer(
    LoadCandidate *candidate, CheckpointLayout *layout)
{
    ModelConfig cfg;

    if (read_config(candidate->stream, &cfg) != 0
        || !checkpoint_layout(cfg, layout)
        || layout->file_bytes != candidate->snapshot_size
        || !snapshot_fits_resident_limit(candidate, layout))
        return -1;

    candidate->tokenizer = tokenizer_read(candidate->stream);
    if (candidate->tokenizer == NULL
        || tokenizer_vocab_size(candidate->tokenizer) != cfg.vocab_size)
        return -1;

    candidate->model = model_new(cfg, 0);
    return model_parameter_count(candidate->model)
        == layout->parameter_floats ? 0 : -1;
}

static int load_checkpoint_parameters(LoadCandidate *candidate)
{
    for (int i = 0; i < candidate->model->param_count; i++) {
        if (param_read(candidate->model->params[i],
                       candidate->stream) != 0)
            return -1;
    }
    return 0;
}

static int consume_checkpoint_trailer(LoadCandidate *candidate)
{
    int32_t checksum;

    if (read_i32(candidate->stream, &checksum) != 0
        || fgetc(candidate->stream) != EOF
        || ferror(candidate->stream))
        return -1;

    FILE *stream = candidate->stream;

    candidate->stream = NULL;
    return fclose(stream);
}

static Model *publish_load_candidate(LoadCandidate *candidate,
                                     Tokenizer **tokenizer)
{
    Model *model = candidate->model;

    *tokenizer = candidate->tokenizer;
    candidate->model = NULL;
    candidate->tokenizer = NULL;
    return model;
}

Model *model_load(Tokenizer **tk, const char *path)
{
    if (tk == NULL)
        return NULL;
    *tk = NULL;

    LoadCandidate candidate;
    CheckpointLayout layout;
    Model *model = NULL;

    load_candidate_init(&candidate);
    if (load_snapshot(&candidate, path) != 0
        || load_header_and_tokenizer(&candidate, &layout) != 0
        || load_checkpoint_parameters(&candidate) != 0
        || consume_checkpoint_trailer(&candidate) != 0)
        goto done;
    model = publish_load_candidate(&candidate, tk);

done:
    load_candidate_free(&candidate);
    return model;
}
