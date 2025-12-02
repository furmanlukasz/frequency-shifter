# Contributing to Harmonic Frequency Shifter

Thank you for your interest in contributing! This document explains how to build, test, and release the plugin.

## Development Setup

### Requirements

- **CMake** 3.22+
- **C++20** compiler:
  - macOS: Xcode Command Line Tools or full Xcode
  - Windows: Visual Studio 2022 with C++ workload
  - Linux: GCC 11+ or Clang 14+
- **Git**

### Clone the Repository

```bash
git clone https://github.com/ludzeller/frequency-shifter.git
cd frequency-shifter
```

## Building

### Quick Build (Current Platform)

```bash
cd plugin

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
```

### macOS Universal Binary (Intel + Apple Silicon)

```bash
cd plugin
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="11.0"

cmake --build build --config Release
```

### Windows

```bash
cd plugin
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Build Output

After building, plugins are located in:

```
plugin/build/FrequencyShifter_artefacts/Release/
├── VST3/
│   └── Frequency Shifter.vst3
├── AU/                          # macOS only
│   └── Frequency Shifter.component
└── Standalone/
    └── Frequency Shifter        # Standalone app
```

## Installing for Testing

### macOS

```bash
# VST3
cp -R "plugin/build/FrequencyShifter_artefacts/Release/VST3/Frequency Shifter.vst3" \
  ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -R "plugin/build/FrequencyShifter_artefacts/Release/AU/Frequency Shifter.component" \
  ~/Library/Audio/Plug-Ins/Components/

# Rescan plugins in your DAW
```

### Windows

Copy the VST3 folder to:
```
C:\Program Files\Common Files\VST3\
```

## Project Structure

```
frequency-shifter/
├── plugin/                      # JUCE plugin source
│   ├── CMakeLists.txt          # Build configuration
│   └── src/
│       ├── PluginProcessor.cpp/h   # Audio processing core
│       ├── PluginEditor.cpp/h      # GUI
│       └── dsp/                    # DSP algorithms
│           ├── STFT.cpp/h
│           ├── PhaseVocoder.cpp/h
│           ├── FrequencyShifter.cpp/h
│           ├── MusicalQuantizer.cpp/h
│           └── Scales.h
├── website/                     # GitHub Pages content
├── docs/                        # Technical documentation
├── legacy/                      # Python prototype (reference)
└── .github/workflows/           # CI/CD pipelines
```

## CI/CD Workflows

### Automatic Builds (`build.yml`)

Every push to `main` and every pull request triggers:
- macOS build (VST3 + AU, universal binary)
- Windows build (VST3)
- Artifacts uploaded for download

### GitHub Pages (`pages.yml`)

Automatically deploys the `website/` folder to GitHub Pages when:
- Changes are pushed to `main` in the `website/` folder
- Workflow is manually triggered

### Releases (`release.yml`)

Triggered by:
- Pushing a version tag (e.g., `v1.0.0`)
- Manual workflow dispatch with version input

## Creating a Release

### 1. Update Version Numbers

Edit `plugin/CMakeLists.txt`:
```cmake
project(FrequencyShifter VERSION 1.0.0)
```

### 2. Commit Changes

```bash
git add -A
git commit -m "Bump version to 1.0.0"
git push origin main
```

### 3. Create and Push Tag

```bash
git tag v1.0.0
git push origin v1.0.0
```

### 4. Automatic Release

The release workflow will:
1. Build plugins for all platforms
2. Create zip archives
3. Create a GitHub Release with:
   - `FrequencyShifter-macOS-VST3.zip`
   - `FrequencyShifter-macOS-AU.zip`
   - `FrequencyShifter-Windows-VST3.zip`

### Manual Release (Alternative)

You can also trigger a release manually:
1. Go to Actions → Release → Run workflow
2. Enter the version (e.g., `v1.0.0`)
3. Click "Run workflow"

## Code Style

### C++

- Use C++20 features where appropriate
- Follow JUCE coding conventions
- Use `jassert()` for debug assertions
- Prefer `const` and `constexpr` where possible
- Use `auto` judiciously

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `camelCase`
- Member variables: `camelCase`
- Constants: `UPPER_SNAKE_CASE`
- Parameters: `camelCase`

## Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Test locally
5. Push and create a PR

### PR Checklist

- [ ] Code compiles without warnings
- [ ] Tested on at least one platform
- [ ] No breaking changes to existing functionality
- [ ] Documentation updated if needed

## Reporting Issues

Please include:
- Platform (macOS/Windows) and version
- DAW name and version
- Steps to reproduce
- Expected vs actual behavior
- Any error messages or crashes

## Architecture Notes

### Audio Processing Flow

```
processBlock()
    ↓
For each channel:
    ↓
STFT::process()
    ├── Forward FFT
    ├── PhaseVocoder::processFrame()
    │       ├── Peak detection
    │       ├── Instantaneous frequency
    │       └── Phase locking
    ├── FrequencyShifter::shiftSpectrum()
    ├── MusicalQuantizer::quantizeToScale()
    └── Inverse FFT (overlap-add)
    ↓
Dry/wet mix
```

### Thread Safety

- Audio processing happens on the audio thread
- GUI updates happen on the message thread
- Use `juce::SpinLock` for audio-thread-safe data sharing
- Parameters are thread-safe via `AudioProcessorValueTreeState`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
