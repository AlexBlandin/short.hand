/*
  cwd_polyglot.c - Cross-platform Current Working Directory Utility

  COMPATIBILITY:
  - C:   C89, C90, C99, C11, C17, C23
  - C++: C++98, C++11, C++14, C++17, C++20, C++23

  USAGE:

  AS A STANDALONE PROGRAM:
     Compile directly.
     $ gcc cwd_polyglot.c -o cwd

     Run:
     $ ./cwd              # Prints /path/to/current/dir
     $ ./cwd -v           # Prints Current directory: /path/to/current/dir
     $ ./cwd --version    # Prints version and build info

  AS A SINGLE FILE HEADER LIBRARY:
     Just include it!

     #include "cwd_polyglot.c"

     The file automatically detects it is being included on GCC/Clang and
     switches to library mode (static functions, no main). Otherwise, you
     need to #define CWD_LIBRARY before #include "cwd_polyglot.c". If you
     just want the header file, use #include "cwd_polyglot.h" instead, as
     that skips the standalone main() function, which demonstrates usage.

  Copyright (C) 2025 by Alex Blandin

  Permission to use, copy, modify, and/or distribute this software for
  any purpose with or without fee is hereby granted.

  THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
  WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
  OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
  FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
  OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#define CWD_VERSION "1.0.0"

/* Detect if used as a library */
#if !defined(CWD_LIBRARY)
  #if defined(__INCLUDE_LEVEL__) && __INCLUDE_LEVEL__ > 0
    #define CWD_LIBRARY
  #endif
#endif

#ifndef CWD_POLYGLOT_H
  #define CWD_POLYGLOT_H

/* Feature Detection & Includes */
  #ifdef __cplusplus
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <iostream>
    #include <string>

    #if defined(__has_include)
      #if __has_include(<cerrno>)
        #include <cerrno>
        #define CWD_HAS_ERRNO 1
      #elif __has_include(<errno.h>)
        #include <errno.h>
        #define CWD_HAS_ERRNO 1
      #endif
    #else
      #include <cerrno>
      #define CWD_HAS_ERRNO 1
    #endif

    #if defined(__cpp_lib_filesystem)                     \
      || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)   \
      || (defined(__cplusplus) && __cplusplus >= 201703L)
      #if __has_include(<filesystem>)
        #include <filesystem>
        #define CWD_USE_STD_FS 1
namespace fs = std::filesystem;
      #endif
    #endif

  #else /* C Mode */
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    #if defined(__has_include)
      #if __has_include(<errno.h>)
        #include <errno.h>
        #define CWD_HAS_ERRNO 1
      #endif
    #else
      #include <errno.h>
      #define CWD_HAS_ERRNO 1
    #endif
  #endif

/* Platform Detection */
  #if defined(_WIN32) || defined(_WIN64)
    #define CWD_PLATFORM_WIN 1
    #ifndef WIN32_LEAN_AND_MEAN
      #define WIN32_LEAN_AND_MEAN
    #endif

    #if defined(__has_include)
      #if __has_include(<windows.h>)
        #include <windows.h>
        #define CWD_HAS_WINDOWS_H 1
      #endif
    #elif defined(_MSC_VER) || defined(__MINGW32__)
      #include <windows.h>
      #define CWD_HAS_WINDOWS_H 1
    #endif

    #include <direct.h> /* _getcwd */

  #else /* POSIX */
    #define CWD_PLATFORM_WIN 0

    #if defined(__cplusplus) && defined(__has_include)
      #if __has_include(<unistd.h>)
        #include <unistd.h>
      #endif
    #else
      #include <unistd.h>
    #endif

    #include <limits.h>
  #endif

  /* Linkage */
  #if defined(CWD_LIBRARY)
    #define CWD_API static
  #else
    #define CWD_API
  #endif

/* Declarations */

  #ifdef __cplusplus
extern "C" {
  #endif

/* Returns heap-allocated UTF-8 string. Caller must free(). */
CWD_API char * cwd_get(void);

/* Prints build info to stdout. */
CWD_API void cwd_info(void);

  #ifdef __cplusplus
}
  #endif

/* C++ Facade */
  #ifdef __cplusplus
namespace cwd {
  inline std::string get_string() {
    /* C++17 std::filesystem */
    #if defined(CWD_USE_STD_FS) && CWD_USE_STD_FS
      #if defined(__cpp_exceptions)
    try {
      return fs::current_path().generic_string();
    }
    catch(...) {}
      #else
          /* If exceptions disabled, fall through to C implementation */
      #endif
    #endif

    /* Wrap C Implementation */
    {
      char * raw = ::cwd_get();
      if (!raw) {
        return std::string();
      }
      std::string res(raw);
      std::free(raw);
      return res;
    }
  }

  inline void info() {
    ::cwd_info();
  }
}
  #endif

#endif /* CWD_POLYGLOT_H */

#if !defined(CWD_ONLY_DECLARATIONS)

  #ifdef __cplusplus
extern "C" {
  #endif

CWD_API void cwd_info(void) {
  char const * lang     = "C89/C90";
  char const * compiler = "Unknown";
  char const * plat     = "Unknown";
  char const * mode     = "Standalone";

  #ifdef CWD_LIBRARY
  mode = "Library";
  #endif

  #if defined(_WIN32)
  plat = "Windows";
  #elif defined(__linux__)
  plat = "Linux";
  #elif defined(__APPLE__)
  plat = "macOS";
  #elif defined(__unix__)
  plat = "Unix";
  #endif

  /* Compiler */
  #if defined(__zig__)
    /* Zig 0.11+ dropped the __zig__ macro. Retained for legacy/future support. */
    #if defined(__cplusplus)
  compiler = "Zig C++";
    #else
  compiler = "Zig cc";
    #endif
  #elif defined(__clang__)
    #if defined(__MINGW32__) || defined(__MINGW64__)
  compiler = "Clang (MinGW)";
    #else
  compiler = "Clang";
    #endif
  #elif defined(_MSC_VER)
  compiler = "MSVC";
  #elif defined(__MINGW32__)
  compiler = "MinGW (GCC)";
  #elif defined(__GNUC__)
  compiler = "GCC";
  #endif

  /* Language Version */
  #ifdef __cplusplus
    #if defined(_MSVC_LANG)
      #if _MSVC_LANG >= 202002L
  lang = "C++20/23";
      #elif _MSVC_LANG >= 201703L
  lang = "C++17";
      #elif _MSVC_LANG >= 201103L
  lang = "C++11/14";
      #else
  lang = "C++ (Legacy)";
      #endif
    #else
      #if __cplusplus >= 202002L
  lang = "C++20/23";
      #elif __cplusplus >= 201703L
  lang = "C++17";
      #elif __cplusplus >= 201103L
  lang = "C++11/14";
      #else
  lang = "C++98";
      #endif
    #endif
  #else
    #if defined(__STDC_VERSION__)
      #if __STDC_VERSION__ >= 201112L
  lang = "C11/17/23";
      #elif __STDC_VERSION__ >= 199901L
  lang = "C99";
      #endif
    #endif
  #endif

  printf("  [Build] Language: %s\n", lang);
  printf("  [Build] Platform: %s\n", plat);
  printf("  [Build] Compiler: %s\n", compiler);
  printf("  [Build] Mode:     %s\n", mode);
}

  #if CWD_PLATFORM_WIN

CWD_API char * cwd_get(void) {
    #if defined(CWD_HAS_WINDOWS_H) && CWD_HAS_WINDOWS_H
  DWORD required_len = GetCurrentDirectoryW(0, NULL);
  if (required_len > 0) {
    wchar_t * wbuf = (wchar_t *) malloc(sizeof(wchar_t) * (required_len + 1));
    if (wbuf) {
      if (GetCurrentDirectoryW(required_len, wbuf) != 0) {
        int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, NULL, 0, NULL, NULL);
        if (utf8_len > 0) {
          char * utf8_buf = (char *) malloc((size_t) utf8_len);
          if (utf8_buf) {
            WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, utf8_buf, utf8_len, NULL, NULL);
            free(wbuf);
            return utf8_buf;
          }
        }
      }
      free(wbuf);
    }
  }
    #endif

  {
    char * buf = _getcwd(NULL, 0);
    return buf;
  }
}

  #else /* POSIX */

CWD_API char * cwd_get(void) {
  size_t size = 1024;
  char * buf;
  char * new_buf;

  buf = (char *) malloc(size);
  if (!buf) {
    return NULL;
  }

  while (1) {
    if (getcwd(buf, size) != NULL) {
      return buf;
    }

    /* Blindly resize if errno is missing or ERANGE */
    #if defined(CWD_HAS_ERRNO)
    if (errno != ERANGE) {
      free(buf);
      return NULL;
    }
    #endif

    size *= 2;
    if (size > 1024 * 1024) { /* Arbitrary 1 MiB Limit */
      free(buf);
      return NULL;
    }

    new_buf = (char *) realloc(buf, size);
    if (!new_buf) {
      free(buf);
      return NULL;
    }
    buf = new_buf;
  }
}

  #endif

  #ifdef __cplusplus
} /* extern "C" */
  #endif

#endif /* !CWD_ONLY_DECLARATIONS */

/* Standalone Program Demo */

#if !defined(CWD_LIBRARY)

  #ifdef __cplusplus

int main(int argc, char ** argv) {
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--version") == 0) {
      std::cout << "cwd utility v" << CWD_VERSION << "\n";
      cwd::info();
      return EXIT_SUCCESS;
    } else if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    }
  }

  std::string path = cwd::get_string();
  if (!path.empty()) {
    if (verbose) {
      std::cout << "Current directory: " << path << "\n";
    } else {
      std::cout << path << "\n";
    }
    return EXIT_SUCCESS;
  }

  std::cerr << "Error: Could not determine current directory.\n";
  return EXIT_FAILURE;
}

  #else

int main(int argc, char ** argv) {
  int    verbose = 0;
  int    i;
  char * path;

  for (i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--version") == 0) {
      printf("cwd utility v%s\n", CWD_VERSION);
      cwd_info();
      return 0;
    } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
      verbose = 1;
    }
  }

  path = cwd_get();
  if (path) {
    if (verbose) {
      printf("Current directory: %s\n", path);
    } else {
      printf("%s\n", path);
    }
    free(path);
    return 0;
  }

  fprintf(stderr, "Error: Could not determine current directory.\n");
  return 1;
}

  #endif /* __cplusplus */

#endif /* !CWD_LIBRARY */
