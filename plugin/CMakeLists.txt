cmake_minimum_required(VERSION 3.22)

# Project setup
project(FrequencyShifter VERSION 0.2.3)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# macOS deployment target (for M1 compatibility)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Build universal binary")
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
    GIT_SHALLOW TRUE
)

# Local patch carried against the pinned Visage: its embedded (plugin) WindowMac constructor never
# calls setDpiScale(), unlike the top-level one — so AU hosts that attach the editor view late
# (e.g. Ableton) initialise the canvas at dpiScale 1.0 while child frames lay out at the native 2×,
# scrambling the UI. VST3/Standalone are unaffected. Applied idempotently at populate time so clean
# builds re-apply it automatically and re-running on an already-patched tree is a no-op.
# Re-verify this patch whenever the GIT_TAG SHA below is bumped.
set(VISAGE_AU_HIDPI_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/patches/visage-au-hidpi.patch")

# Second local patch: the embedded (plugin) NSView never reclaims the window's first
# responder on click, so when a host holds keyboard focus, typing (e.g. naming a preset)
# never reaches the Visage UI — the preset-name field appears focused but swallows no keys.
# Make the view first responder on mouseDown. Same idempotent apply as above; re-verify
# whenever the Visage GIT_TAG SHA below is bumped.
set(VISAGE_KEYFOCUS_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/patches/visage-plugin-keyfocus.patch")

# Both patches modify visage's macOS-only windowing_macos.mm, which isn't compiled on
# Windows/Linux. Applying these LF patches against a fresh (CRLF) Windows checkout fails
# the configure step, so only patch on Apple; elsewhere the patch step is a no-op.
# (Combined into one git apply call so the command has no ';' — CMake would otherwise
# split the PATCH_COMMAND list on it.)
if(APPLE)
    set(VISAGE_PATCH_COMMAND sh -c "git apply --reverse --check '${VISAGE_AU_HIDPI_PATCH}' '${VISAGE_KEYFOCUS_PATCH}' 2>/dev/null && echo 'visage patches: already applied' || git apply --verbose '${VISAGE_AU_HIDPI_PATCH}' '${VISAGE_KEYFOCUS_PATCH}'")
else()
    set(VISAGE_PATCH_COMMAND "${CMAKE_COMMAND}" -E echo "Skipping macOS-only Visage patches on this platform")
endif()

FetchContent_Declare(
    visage
    GIT_REPOSITORY https://github.com/VitalAudio/visage.git
    # Pinned: tracking `main` made CI fragile to upstream Visage breakage.
    # Bump this SHA deliberately after testing against the new revision.
    GIT_TAG b0b2ee89abb3f1173dad4d80460197eb96afa724
    GIT_SHALLOW FALSE  # SHA pins require full history (GIT_SHALLOW only works with tags/branches)
    PATCH_COMMAND ${VISAGE_PATCH_COMMAND}
)

FetchContent_MakeAvailable(JUCE visage)

# --- Embed UI fonts (Inter + IBM Plex Mono) ---------------------------------
# Visage's resource embedder (add_embedded_resources) is available once visage is
# made available. Generates holy::fonts::Inter_*_ttf / IBMPlexMono_Light_ttf as
# visage::EmbeddedFile, compiled into the binary (no runtime font dependency).
file(GLOB HOLY_FONT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/fonts/*.ttf")
add_embedded_resources(HolyShifterFonts "holy_fonts.h" "holy::fonts" "${HOLY_FONT_FILES}")

# Embed UI images (e.g. the Heathen Machines logo) the same way → holy::images::*_png,
# decoded + atlased by Visage at draw time (canvas.image()).
file(GLOB HOLY_IMAGE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/*.png")
add_embedded_resources(HolyShifterImages "holy_images.h" "holy::images" "${HOLY_IMAGE_FILES}")

# Plugin sources
set(PLUGIN_SOURCES
    src/PluginProcessor.cpp
    src/PluginProcessor.h
    src/VisageHostEditor.cpp
    src/VisageHostEditor.h
    src/PresetManager.cpp
    src/PresetManager.h
    # Visage UI
    src/ui/HolyTheme.h
    src/ui/VisageParamAttachment.h
    src/ui/HolyShifterUI.cpp
    src/ui/HolyShifterUI.h
    src/ui/controls/VisageControl.h
    src/ui/controls/HolyRotaryKnob.cpp
    src/ui/controls/HolyRotaryKnob.h
    src/ui/controls/HolySlider.cpp
    src/ui/controls/HolySlider.h
    src/ui/controls/HolyToggle.cpp
    src/ui/controls/HolyToggle.h
    src/ui/controls/HolyComboBox.cpp
    src/ui/controls/HolyComboBox.h
    src/ui/controls/HolyPianoKeyboard.cpp
    src/ui/controls/HolyPianoKeyboard.h
    src/ui/controls/HolySegmentedControl.cpp
    src/ui/controls/HolySegmentedControl.h
    # DSP modules
    src/dsp/STFT.cpp
    src/dsp/STFT.h
    src/dsp/MusicalQuantizer.cpp
    src/dsp/MusicalQuantizer.h
    src/dsp/Scales.h
)

# Add the plugin target
juce_add_plugin(FrequencyShifter
    # Plugin info
    COMPANY_NAME "Heathen Machines"
    BUNDLE_ID "com.harmonictools.frequencyshifter"
    PLUGIN_MANUFACTURER_CODE Hrmt
    PLUGIN_CODE Fshf

    # Plugin formats to build
    FORMATS AU VST3 AUv3 Standalone

    # juce_vst3_helper can't load this bundle to emit moduleinfo.json ("couldn't be loaded").
    # This is a SEPARATE, pre-existing issue from the automation hang (which is fixed): the
    # VST3-SDK Module loader rejects the bundle although JUCE hosts (pluginval / pedalboard /
    # DAWs) load it fine. Skip the (optional) manifest so the VST3 builds; DAWs load the
    # plugin without moduleinfo.json — it is only a host-scan speed optimisation.
    VST3_AUTO_MANIFEST FALSE

    # Plugin name. Single source of truth for the version is project(... VERSION) above;
    # it flows into JucePlugin_VersionString and holy::kVersion (Version.h).
    # Note for devs: dropping the version suffix means DAWs no longer auto-rescan on
    # iteration. Bump PROJECT_VERSION for a fresh rescan, or clear the host plugin cache.
    PRODUCT_NAME "Holy Shifter"
    VERSION ${PROJECT_VERSION}

    # Plugin characteristics
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE

    # iOS/AUv3 settings
    PLUGIN_AU_EXPORT_PREFIX FreqShifterAU
    AU_MAIN_TYPE kAudioUnitType_Effect

    # Copy plugin to system folder after build (for testing)
    COPY_PLUGIN_AFTER_BUILD TRUE
)

# Generate JUCE header
juce_generate_juce_header(FrequencyShifter)

# Generate Version.h from PROJECT_VERSION for in-code access (About dialogs, telemetry, etc.)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/Version.h
    @ONLY
)
target_include_directories(FrequencyShifter PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Add source files
target_sources(FrequencyShifter PRIVATE ${PLUGIN_SOURCES})

# Compile definitions
target_compile_definitions(FrequencyShifter
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

# Link JUCE modules and Visage
target_link_libraries(FrequencyShifter
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        visage
        HolyShifterFonts
        HolyShifterImages
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Enable all warnings
if(MSVC)
    target_compile_options(FrequencyShifter PRIVATE /W4)
else()
    target_compile_options(FrequencyShifter PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# C++ DSP unit tests (Catch2). Off by default; enable with:
#   cmake -B build -DHOLY_SHIFTER_BUILD_TESTS=ON
#   cmake --build build --target dsp_tests && build/tests/dsp/dsp_tests
# DSP modules under src/dsp/ are pure C++ (no JUCE), so they link standalone.
# ---------------------------------------------------------------------------
option(HOLY_SHIFTER_BUILD_TESTS "Build C++ DSP unit tests" OFF)

if(HOLY_SHIFTER_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.6.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    enable_testing()
    add_subdirectory(tests/dsp)
endif()
