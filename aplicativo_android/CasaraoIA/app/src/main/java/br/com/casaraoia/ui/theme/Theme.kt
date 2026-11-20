package br.com.casaraoia.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColorScheme = lightColorScheme(
    primary = ColonialBlue,
    onPrimary = WarmSurface,
    primaryContainer = LightBlueContainer,
    secondary = TileBlue,
    tertiary = HeritageGold,
    background = WarmBackground,
    onBackground = Graphite,
    surface = WarmSurface,
    onSurface = Graphite
)

private val DarkColorScheme = darkColorScheme(
    primary = LightBlue,
    secondary = TileBlue,
    tertiary = HeritageGold
)

@Composable
fun CasaraoIATheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = if (darkTheme) {
            DarkColorScheme
        } else {
            LightColorScheme
        },
        typography = Typography,
        content = content
    )
}
