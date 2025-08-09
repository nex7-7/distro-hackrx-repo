# File Security and Robust Handling Implementation

## Overview

This document outlines the comprehensive security improvements implemented to handle problematic files robustly, including recursive ZIPs, large files, and blacklisted file types. When these conditions are detected, the system responds with "Files Not found" to all queries instead of failing.

## Security Features Implemented

### 1. File Size Limits

**Configuration:**
- Maximum file size: 50MB (configurable via `MAX_FILE_SIZE_BYTES`)
- Applies to both downloaded and local files
- Checked during download (streaming) and before processing

**Implementation:**
- Size checking during download with streaming to prevent memory issues
- Pre-processing validation for local files
- Clear error messages with actual vs. allowed file sizes

**Behavior:**
- Files exceeding the limit trigger "Files Not found" response
- Logging includes actual file size and limit exceeded

### 2. Blacklisted File Extensions

**Blocked Extensions:**
```python
BLACKLISTED_EXTENSIONS = {
    "bin", "exe", "dll", "so", "dylib", "app", "deb", "rpm",
    "msi", "dmg", "pkg", "run", "tar", "gz", "bz2", "xz", "7z", "rar"
}
```

**Rationale:**
- **bin**: Binary files that may contain malicious code
- **exe, dll, so, dylib**: Executable files and libraries
- **msi, dmg, pkg, deb, rpm**: Installation packages
- **tar, gz, bz2, xz, 7z, rar**: Compressed archives (except ZIP which is handled specially)

**Implementation:**
- Extension checking during download (before downloading)
- Extension checking before file processing
- Case-insensitive matching

### 3. Recursive ZIP Protection

**Configuration:**
- Maximum ZIP extraction depth: 1 level (`MAX_ZIP_DEPTH = 1`)
- Prevents ZIP bombs and deeply nested archive attacks

**Protection Mechanisms:**
1. **Depth Tracking**: Each ZIP extraction increments depth counter
2. **Nested ZIP Detection**: Identifies ZIP files within ZIP files
3. **Early Termination**: Stops processing when depth limit reached
4. **Safe Member Filtering**: Skips suspicious files and hidden/system files

**Implementation Details:**
- ZIP depth parameter passed through recursive calls
- Nested ZIPs beyond depth limit are skipped (not extracted)
- Hidden files (starting with `.`) and macOS metadata files are filtered out
- `__MACOSX` folders are automatically skipped

### 4. Content Validation

**Multi-Layer Validation:**
1. **Initial Ingestion**: Checks if any content was extracted
2. **Post-Semantic Chunking**: Verifies content remains after processing
3. **Parser Validation**: Ensures individual parsers return meaningful content

**Empty Content Scenarios:**
- Empty ZIP files
- ZIP files containing only blacklisted files
- Corrupted or unreadable files
- Files with no extractable text content

## Response Behavior

### "Files Not found" Triggers

The system returns "Files Not found" for all questions when:

1. **File too large** (>50MB)
2. **Blacklisted file extension** (any in `BLACKLISTED_EXTENSIONS`)
3. **Recursive ZIP depth exceeded** (>1 level deep)
4. **Empty or invalid ZIP files**
5. **ZIP containing only blacklisted files**
6. **Corrupted files** that cannot be processed
7. **No extractable content** after processing

### Implementation in main.py

```python
# Check if any valid content was found
if not chunks:
    # Return "Files Not found" for all questions
    answers = ["Files Not found"] * len(questions)
    return QueryResponse(answers=answers)
```

### Logging and Monitoring

**Security Events Logged:**
- File security violations with details
- ZIP depth violations
- Blacklisted file attempts
- Large file rejections
- Content extraction failures

**Log Fields Include:**
- File path and extension
- Security violation type
- Actual vs. allowed values (size, depth)
- Processing context (ZIP depth, source URL)

## Error Handling Strategy

### HTTPException Mapping

| Security Violation | HTTP Status | Response |
|-------------------|-------------|----------|
| File too large | 413 | "Files Not found" |
| Blacklisted extension | 415 | "Files Not found" |
| Recursive ZIP | 422 | "Files Not found" |
| Invalid ZIP | 422 | "Files Not found" |
| Processing error | 500 | "Files Not found" |

### Graceful Degradation

1. **Security violations** → "Files Not found" response
2. **Processing errors** → "Files Not found" response  
3. **Empty content** → "Files Not found" response
4. **Network errors** → Standard HTTP error response

## Testing

### Test Files Created

1. **Large File Test**: 60MB file to test size limits
2. **Blacklisted Extension Test**: .bin, .exe, .dll files
3. **Recursive ZIP Test**: ZIP containing ZIP files
4. **Empty ZIP Test**: ZIP with no content
5. **Blacklisted ZIP Test**: ZIP containing only blacklisted files
6. **Valid File Test**: Small text file for comparison

### Test Scenarios

```python
# Test cases that should return "Files Not found"
test_cases = [
    ('large_file', 'Large File (>50MB)'),
    ('bin_file', 'Blacklisted File (.bin)'),
    ('recursive_zip', 'Recursive ZIP'),
    ('empty_zip', 'Empty ZIP'),
    ('blacklisted_zip', 'ZIP with Blacklisted Files'),
]
```

## Configuration

### Adjustable Parameters

```python
# File size limit (in bytes)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

# ZIP extraction depth limit
MAX_ZIP_DEPTH = 1

# Blacklisted extensions (set)
BLACKLISTED_EXTENSIONS = {
    "bin", "exe", "dll", # ... add more as needed
}
```

### Environment Variables

These could be made configurable via environment variables:
- `MAX_FILE_SIZE_MB`: Maximum file size in MB
- `MAX_ZIP_DEPTH`: Maximum ZIP extraction depth
- `BLACKLISTED_EXTENSIONS`: Comma-separated list of blocked extensions

## Security Benefits

1. **Prevents ZIP Bombs**: Limits extraction depth and file sizes
2. **Blocks Malicious Files**: Rejects potentially dangerous file types
3. **Resource Protection**: Prevents excessive memory/disk usage
4. **Attack Surface Reduction**: Minimizes file types that can be processed
5. **Consistent Responses**: Uniform "Files Not found" for security violations
6. **Audit Trail**: Comprehensive logging of all security events

## Performance Considerations

1. **Streaming Downloads**: Large files are rejected during download, not after
2. **Early Validation**: Security checks happen before expensive processing
3. **Memory Efficient**: ZIP extraction uses temporary directories with cleanup
4. **Depth Limiting**: Prevents infinite recursion in ZIP processing

## Future Enhancements

1. **Virus Scanning Integration**: Add virus scanning for uploaded files
2. **Content-Type Validation**: Verify MIME types match file extensions
3. **Rate Limiting**: Implement rate limiting for file uploads
4. **Quarantine System**: Store suspicious files for analysis
5. **Admin Dashboard**: Interface for monitoring security events
