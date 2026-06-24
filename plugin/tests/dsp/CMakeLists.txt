# Standalone C++ DSP test binary. Links DSP sources directly (no JUCE/Visage)
# so the test cycle is fast and CI can run it on any platform without a host SDK.

add_executable(dsp_tests
    STFT_test.cpp
    MusicalQuantizer_test.cpp
    Robustness_test.cpp
    HilbertShifter_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/dsp/STFT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/dsp/MusicalQuantizer.cpp
)

target_include_directories(dsp_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(dsp_tests PRIVATE Catch2::Catch2WithMain)

# STFT.cpp uses Apple vDSP (Accelerate) for the FFT on Apple platforms.
if(APPLE)
    target_link_libraries(dsp_tests PRIVATE "-framework Accelerate")
endif()

# Match the main target's C++ standard.
target_compile_features(dsp_tests PRIVATE cxx_std_20)

# Catch2 v3 ships its own discovery helper.
include(${catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(dsp_tests)
