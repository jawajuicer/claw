package com.claw.assistant.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat

// Claw brand colors matching the PWA dark theme
val ClawBackground = Color(0xFF0D1117)
val ClawSurface = Color(0xFF161B22)
val ClawSurfaceVariant = Color(0xFF21262D)
val ClawAccent = Color(0xFF58A6FF)
val ClawAccentDim = Color(0xFF388BFD)
val ClawTextPrimary = Color(0xFFE6EDF3)
val ClawTextSecondary = Color(0xFF8B949E)
val ClawError = Color(0xFFF85149)
val ClawSuccess = Color(0xFF3FB950)
val ClawWarning = Color(0xFFD29922)
val ClawBorder = Color(0xFF30363D)
val ClawUserBubble = Color(0xFF1F6FEB)
val ClawAssistantBubble = Color(0xFF21262D)

private val ClawColorScheme = darkColorScheme(
    primary = ClawAccent,
    onPrimary = ClawBackground,
    primaryContainer = ClawAccentDim,
    onPrimaryContainer = ClawTextPrimary,
    secondary = ClawAccent,
    onSecondary = ClawBackground,
    secondaryContainer = ClawSurfaceVariant,
    onSecondaryContainer = ClawTextPrimary,
    tertiary = ClawSuccess,
    onTertiary = ClawBackground,
    background = ClawBackground,
    onBackground = ClawTextPrimary,
    surface = ClawSurface,
    onSurface = ClawTextPrimary,
    surfaceVariant = ClawSurfaceVariant,
    onSurfaceVariant = ClawTextSecondary,
    error = ClawError,
    onError = ClawTextPrimary,
    outline = ClawBorder,
    outlineVariant = ClawBorder,
)

private val ClawTypography = Typography(
    headlineLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 28.sp,
        lineHeight = 36.sp,
        color = ClawTextPrimary
    ),
    headlineMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 22.sp,
        lineHeight = 28.sp,
        color = ClawTextPrimary
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 20.sp,
        lineHeight = 26.sp,
        color = ClawTextPrimary
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 16.sp,
        lineHeight = 22.sp,
        color = ClawTextPrimary
    ),
    bodyLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        color = ClawTextPrimary
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        color = ClawTextPrimary
    ),
    bodySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
        lineHeight = 16.sp,
        color = ClawTextSecondary
    ),
    labelLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        color = ClawTextPrimary
    ),
    labelSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        lineHeight = 16.sp,
        color = ClawTextSecondary
    ),
)

@Composable
fun ClawTheme(content: @Composable () -> Unit) {
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = ClawBackground.toArgb()
            window.navigationBarColor = ClawBackground.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = false
            WindowCompat.getInsetsController(window, view).isAppearanceLightNavigationBars = false
        }
    }

    MaterialTheme(
        colorScheme = ClawColorScheme,
        typography = ClawTypography,
        content = content
    )
}
