# ET: Legacy – Copilot Coding Agent Instructions

## Project Overview

ET: Legacy is an open-source project based on the GPL release of *Wolfenstein: Enemy Territory*. It has two major components:

1. **Engine** (`etl` / `etlded`) – A modernised C/C++ game engine that fixes bugs, removes outdated dependencies, and adds features while remaining wire-compatible with the original ET 2.60b.
2. **Legacy mod** – Server/client game-logic DLLs (`.so`/`.dll`) written in C that add gameplay features and are extensible via Lua scripts.

Current version: see `VERSION.txt` (`VERSION_MAJOR`, `VERSION_MINOR`, `VERSION_PATCH`).

---

## Repository Layout

```
etlegacy/
├── src/                  # All C/C++ source code
│   ├── botlib/           # Bot AI library
│   ├── cgame/            # Client-side mod code (cg_*.c)
│   ├── client/           # Engine client (cl_*.c)
│   ├── db/               # SQLite database layer
│   ├── game/             # Server-side mod code (g_*.c, bg_*.c)
│   ├── irc/              # IRC client
│   ├── null/             # Null/stub implementations
│   ├── qcommon/          # Shared engine utilities (q_shared.h is included everywhere)
│   ├── renderer/         # OpenGL 1/2 renderer (renderer1)
│   ├── renderer2/        # OpenGL 3 renderer (renderer2)
│   ├── rendererGLES/     # OpenGL ES renderer
│   ├── renderer_vk/      # Vulkan renderer
│   ├── renderercommon/   # Renderer-independent helpers
│   ├── sdl/              # SDL2 platform backend
│   ├── server/           # Dedicated server code (sv_*.c)
│   ├── sys/              # OS-specific code
│   ├── tvgame/           # TV-game (spectator) support
│   └── ui/               # In-game UI menus
├── cmake/                # CMake modules (ETLBuild*.cmake, Find*.cmake, etc.)
├── etmain/               # Shipped game scripts, configs, assets
├── libs/                 # Bundled third-party libraries (git submodule → etlegacy-libs)
├── misc/                 # Developer tooling scripts (Python)
├── vendor/               # Vendored source (Lua, luasql, …)
├── app/                  # Android application wrapper
├── docs/                 # Developer documentation (Markdown)
├── CMakeLists.txt        # Top-level CMake project
├── easybuild.sh          # Convenience build wrapper (Linux/macOS)
├── easybuild.bat         # Convenience build wrapper (Windows)
├── pixi.toml             # Pixi environment for developer tooling
├── uncrustify.cfg        # C/C++ formatter configuration
└── .editorconfig         # Editor whitespace rules
```

---

## Building

### Quick start (recommended)

```sh
# Linux / macOS – 64-bit
./easybuild.sh -64

# Linux / macOS – 32-bit (legacy compatibility)
./easybuild.sh

# Windows
easybuild.bat
```

`easybuild.sh` handles CMake configuration, compilation, and installation to `~/etlegacy` (or `%USERPROFILE%\Documents\ETLegacy-Build` on Windows).

### Manual CMake build

```sh
mkdir build && cd build
cmake ..          # add -DBUNDLED_LIBS=YES if system libs are missing
make -j$(nproc)
```

Important CMake booleans (all `ON` by default unless noted):

| Variable | Purpose |
|---|---|
| `BUILD_CLIENT` | Build the graphical client |
| `BUILD_SERVER` | Build the dedicated server |
| `BUILD_MOD` | Build the Legacy mod DLLs |
| `BUNDLED_LIBS` | Use vendored libraries from `libs/` submodule |
| `CROSS_COMPILE32` | Cross-compile 32-bit binary (auto-disabled on Apple/OpenBSD) |
| `FEATURE_RENDERER1` | Classic OpenGL 1/2 renderer |
| `FEATURE_RENDERER2` | Extended OpenGL 3 renderer (default OFF) |
| `FEATURE_RENDERER_VULKAN` | Vulkan renderer (default OFF) |
| `FEATURE_LUA` | Lua scripting support in mod |
| `FEATURE_LUAJIT` | Use LuaJIT instead of PUC Lua (default OFF) |

### Submodules

Bundled libraries live in `libs/` (a shallow submodule pointing to `etlegacy/etlegacy-libs`). If they are missing:

```sh
git submodule init
git submodule update
```

---

## Code Style

### Indentation & whitespace

- **C/C++ files**: tabs, displayed as 4 spaces. Never use spaces for indentation in `.c`/`.h` files.
- **JSON / YAML**: 2-space indent.
- **Python / shell scripts**: 4-space indent.
- **Markdown**: trailing whitespace is allowed (for line breaks).
- All files must end with a newline.
- Windows line endings (`\r\n`) only for `.bat` files; everything else uses `\n`.

These rules are enforced by `.editorconfig`.

### C formatter – Uncrustify

Run `uncrustify` (configured by `uncrustify.cfg`) before committing C/C++ changes. The convenience wrapper is:

```sh
# Using pixi (preferred):
pixi run autoformat

# Or call uncrustify directly:
uncrustify -c uncrustify.cfg --replace <file.c>
```

The CI `pre-build` job runs `pixi run check-changes` which will fail if code is not properly formatted.

### Python formatter – Black

All Python files in `misc/` must be formatted with `black`:

```sh
pixi run autoformat   # formats Python files too
```

### File header

Every C/C++ source and header file **must** begin with the GPL copyright block followed by a Doxygen `@file` doc-comment, for example:

```c
/*
 * Wolfenstein: Enemy Territory GPL Source Code
 * Copyright (C) 1999-2010 id Software LLC, a ZeniMax Media company.
 *
 * ET: Legacy
 * Copyright (C) 2012-2024 ET:Legacy team <mail@etlegacy.com>
 *
 * This file is part of ET: Legacy - http://www.etlegacy.com
 * ...
 */
/**
 * @file my_file.c
 * @brief Short description of what this file does
 */
```

Use existing files (e.g. `src/cgame/cg_main.c`, `src/game/g_main.c`) as templates.

---

## Key Conventions

### Backward compatibility with ET 2.60b

Structs and enums annotated with `ETL_260B_NOEDIT` (defined in `src/qcommon/q_shared.h`) **must not be reordered or have fields removed**. Fields may be added at the end if the macro is `ETL_260B_NOEDIT_EXTEND`. Violating this breaks compatibility with original 2.60b clients and servers.

### Feature flags

Optional features are guarded by `#ifdef FEATURE_<NAME>` preprocessor defines that correspond to CMake `FEATURE_<NAME>` options. Never add feature code without the corresponding `#ifdef` guard.

### Doxygen comments

Use `/** … */` blocks with `@param`, `@return`, `@brief`, and `@note` tags for public API functions. Single-line comments inside function bodies use `//`.

### Naming

- Prefixes mirror the directory: `CG_` for `src/cgame/`, `G_` for `src/game/`, `SV_` for `src/server/`, `R_` for renderers, `COM_` / `Cvar_` / `FS_` for `qcommon/`, etc.
- Types: `typedef struct` or `typedef enum`, typically followed by a `_t` suffix or a `_s` tag.

---

## CI / Automated Checks

Workflows are in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PRs to `master`, pushes to `master` | Format checks + compile matrix |
| `build.yml` | Version tags (`v*`), weekly schedule, manual | Full release builds |
| `coverity-scan.yml` | Schedule | Static analysis |

The CI `pre-build` job runs `pixi run check-changes --github "origin/master"` which validates formatting (uncrustify, black, shfmt, prettier, actionlint). **All checks must pass before merge.**

---

## Pixi Developer Tooling

[Pixi](https://prefix.dev/) manages the Python/tool environment declared in `pixi.toml`. Install pixi, then:

```sh
pixi run autoformat        # format all changed files
pixi run check-changes     # validate formatting (same as CI)
```

Pixi environments:
- Default – git, python, requests
- `validation` – adds uncrustify, black, shfmt, prettier, shellcheck, actionlint
- `libclang` – adds python-clang for Lua API binding generation

---

## Running / Testing

ET: Legacy does not have a standalone unit-test suite. Validation is done by:

1. **Successful compilation** across platforms (Linux 32/64-bit, Windows, macOS, ARM/aarch64).
2. **CI format checks** (`pixi run check-changes`).
3. **Runtime testing** – copy `pak0.pk3` (from the original free ET release) into `etmain/` and launch the built binary.

---

## Common Errors and Workarounds

| Error | Cause | Fix |
|---|---|---|
| `submodule` paths missing / `CMake` can't find bundled libs | `libs/` submodule not initialised | `git submodule init && git submodule update` |
| 32-bit build fails on 64-bit Linux (missing `-dev` packages) | 32-bit dev libraries not installed | `export CC="gcc -m32" CXX="g++ -m32"` before cmake, or use `easybuild.sh` which sets this automatically |
| `libcurl` compilation aborts on Windows (missing `sed`) | `sed` not on `PATH` | Install Git for Windows and ensure it is on `PATH` (includes `sed`), or download from GnuWin |
| `pixi` command not found | Pixi not installed | Install from https://prefix.dev/ or via `curl -fsSL https://pixi.sh/install.sh | bash` |
| SDL audio init failure (`dsp: No such audio device`) | Missing PulseAudio/ALSA headers | `apt install libpulse-dev` or `libasound2-dev`, then rebuild |
| Mouse acceleration issues on X11 | Missing XInput | `apt install libxi-dev`, then rebuild |
| Uncrustify format check fails in CI | Code not formatted | Run `pixi run autoformat` locally before pushing |

---

## External Resources

- Wiki / FAQ: https://github.com/etlegacy/etlegacy/wiki
- Coding conventions wiki page: https://github.com/etlegacy/etlegacy/wiki/Coding-Conventions
- How to commit code: https://github.com/etlegacy/etlegacy/wiki/How-to-commit-Your-Code
- Lua API docs: https://etlegacy-lua-docs.readthedocs.io
- Full documentation: https://etlegacy.readthedocs.io
