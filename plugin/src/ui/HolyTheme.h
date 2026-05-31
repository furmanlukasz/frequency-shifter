#pragma once

#include <visage_graphics/theme.h>

// Holy Shifter color palette — dark theme with gold/amber accent
VISAGE_THEME_COLOR(HolyBackground,    0xFF0A0A0C);
VISAGE_THEME_COLOR(HolySurface,       0xFF111113);
VISAGE_THEME_COLOR(HolyStrip,         0xFF0E0E10);
VISAGE_THEME_COLOR(HolyStripBorder,   0xFF1A1A1D);
VISAGE_THEME_COLOR(HolyRaised,        0xFF161618);
VISAGE_THEME_COLOR(HolyBorder,        0xFF1E1E22);
VISAGE_THEME_COLOR(HolyBorderDim,     0xFF151517);
VISAGE_THEME_COLOR(HolyPanelBg,       0xFF0D0D0F);
VISAGE_THEME_COLOR(HolyPanelBorder,   0xFF1C1C20);
VISAGE_THEME_COLOR(HolyModeSelectorBg,0xFF0C0C0E);
VISAGE_THEME_COLOR(HolyMaskBg,        0xFF19191D);
VISAGE_THEME_COLOR(HolyPanelGradTop,  0xFF19191D);
VISAGE_THEME_COLOR(HolyPanelGradBot,  0xFF101013);
VISAGE_THEME_COLOR(HolyMixGradTop,    0xFF0F0F11);

VISAGE_THEME_COLOR(HolyText,          0xFFE8E4DB);
VISAGE_THEME_COLOR(HolyTextSec,       0xFFB8B2A6);  // WCAG AA: ~9:1 on HolyBackground
VISAGE_THEME_COLOR(HolyTextMuted,     0xFF3E3A34);

VISAGE_THEME_COLOR(HolyAccent,        0xFFC9A96E);
VISAGE_THEME_COLOR(HolyAccentDim,     0xFF6B5D3D);
VISAGE_THEME_COLOR(HolyAccentGlow,    0x26C9A96E);
VISAGE_THEME_COLOR(HolyTrack,         0xFF252320);

// Theme value constants
VISAGE_THEME_VALUE(HolyKnobArcWidth,  2.5f);
VISAGE_THEME_VALUE(HolySliderHeight,  1.5f);
VISAGE_THEME_VALUE(HolyThumbSize,     7.0f);
VISAGE_THEME_VALUE(HolyToggleWidth,   30.0f);
VISAGE_THEME_VALUE(HolyToggleHeight,  15.0f);
VISAGE_THEME_VALUE(HolyToggleDotSize, 11.0f);

#include <visage_graphics/font.h>

namespace holy {
    // System font path — using SF Pro on macOS, fallback to Arial elsewhere
    inline visage::Font makeFont(float size)
    {
#if __APPLE__
        return visage::Font(size, "/System/Library/Fonts/Helvetica.ttc");
#elif _WIN32
        return visage::Font(size, "C:\\Windows\\Fonts\\arial.ttf");
#else
        return visage::Font(size, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
#endif
    }

    // Reduce alpha of an ARGB color for dimmed/disabled controls
    inline constexpr unsigned int dimColor(unsigned int color, bool dimmed)
    {
        if (!dimmed) return color;
        unsigned int alpha = (color >> 24) & 0xFF;
        alpha = static_cast<unsigned int>(alpha * 0.25f);
        return (alpha << 24) | (color & 0x00FFFFFFu);
    }

    // Raw color values for use in draw() calls that need direct ARGB
    namespace colors {
        static constexpr unsigned int background  = 0xFF0A0A0C;
        static constexpr unsigned int surface     = 0xFF111113;
        static constexpr unsigned int strip       = 0xFF0E0E10;
        static constexpr unsigned int stripBorder = 0xFF1A1A1D;
        static constexpr unsigned int raised      = 0xFF161618;
        static constexpr unsigned int border      = 0xFF1E1E22;
        static constexpr unsigned int borderDim   = 0xFF151517;
        static constexpr unsigned int panelBg     = 0xFF0D0D0F;
        static constexpr unsigned int panelBorder = 0xFF1C1C20;
        static constexpr unsigned int text        = 0xFFE8E4DB;
        static constexpr unsigned int textSec     = 0xFFB8B2A6;  // WCAG AA: ~9:1 on background
        static constexpr unsigned int textMuted   = 0xFF3E3A34;
        static constexpr unsigned int accent      = 0xFFC9A96E;
        static constexpr unsigned int accentDim   = 0xFF6B5D3D;
        static constexpr unsigned int accentHalf  = 0x80C9A96E; // 50% alpha accent
        static constexpr unsigned int track       = 0xFF252320;
        static constexpr unsigned int modeSelectorBg = 0xFF0C0C0E;
        static constexpr unsigned int maskBg      = 0xFF19191D;
        static constexpr unsigned int panelGradTop= 0xFF19191D;
        static constexpr unsigned int panelGradBot= 0xFF101013;
        static constexpr unsigned int mixGradTop  = 0xFF0F0F11;
    }
}
