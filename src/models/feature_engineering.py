import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

class FeatureEngineering:
    """Clase para calcular indicadores técnicos usando bibliotecas de Python puras"""
    
    def __init__(self):
        """Inicializa la clase de ingeniería de características"""
        pass
    
    def calculate_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calcula todos los indicadores para un DataFrame con datos OHLCV
        
        Args:
            data: DataFrame con columnas 'Open', 'High', 'Low', 'Close', 'Volume'
            timeframe: Timeframe de los datos ('1d', '1h', '15m')
            
        Returns:
            DataFrame con todos los indicadores calculados
        """
        # Verificar si el DataFrame está vacío
        if data.empty:
            print(f"Warning: DataFrame vacío para timeframe {timeframe}")
            return pd.DataFrame()
            
        # Crear copia para no modificar el original
        df = data.copy()
        
        # Crear indicadores según el timeframe
        if timeframe == '1d':
            df = self._calculate_daily_indicators(df)
        elif timeframe == '1h':
            df = self._calculate_hourly_indicators(df)
        elif timeframe == '15m':
            df = self._calculate_15min_indicators(df)
        else:
            raise ValueError(f"Timeframe no soportado: {timeframe}")
        
        # En lugar de eliminar filas con NaN, rellenarlos de manera apropiada
        
        # 1. Para indicadores de tendencia (EMAs, SMAs, MACD, etc.), usar forward fill y luego backward fill
        trend_indicators = [col for col in df.columns if any(ind in col.lower() for ind in 
                                                        ['ema', 'sma', 'macd', 'bb_', 'adx', 'obv', 'market_phase'])]
        for col in trend_indicators:
            if col in df.columns:
                # Forward fill primero, luego backward fill para los NaN restantes
                df[col] = df[col].ffill().bfill()
        
        # 2. Para osciladores (RSI, Estocástico, etc.), rellenar con valores neutrales
        oscillator_indicators = [col for col in df.columns if any(ind in col.lower() for ind in 
                                                                ['rsi', 'stoch', 'momentum', 'divergence'])]
        for col in oscillator_indicators:
            if col in df.columns:
                # Para RSI y Estocástico, 50 es un valor neutral
                if 'rsi' in col.lower():
                    df[col] = df[col].fillna(50.0)
                # Para otros osciladores, usar 0 como valor neutral
                else:
                    df[col] = df[col].fillna(0.0)
        
        # 3. Para indicadores de volatilidad (ATR, etc.), usar el valor medio
        volatility_indicators = [col for col in df.columns if any(ind in col.lower() for ind in 
                                                            ['atr', 'volatility', 'squeeze'])]
        for col in volatility_indicators:
            if col in df.columns:
                # Usar la media del indicador si hay suficientes valores no-NaN
                if df[col].count() > 5:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                # Si no hay suficientes valores, usar 0
                else:
                    df[col] = df[col].fillna(0.0)
        
        # 4. Para cualquier columna restante con NaN, usar 0
        remaining_nan_cols = df.columns[df.isna().any()].tolist()
        for col in remaining_nan_cols:
            df[col] = df[col].fillna(0.0)
        
        # Verificar si aún hay NaN después del procesamiento
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            print(f"Warning: Aún hay valores NaN en las columnas {nan_cols} para timeframe {timeframe}")
            # Último recurso: llenar todos los NaN restantes con 0
            df = df.fillna(0.0)
        
        # Verificar integridad final
        print(f"Indicadores calculados para timeframe {timeframe}: {len(df)} filas, {len(df.columns)} columnas")
        
        return df
    
    def _calculate_daily_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para el timeframe diario"""
        # SMA 20 días y su pendiente
        df['SMA20'] = self._calculate_sma(df['Close'], 20)
        df['SMA20_slope'] = self._calculate_slope(df['SMA20'], 5)
        
        # Fase del mercado (tendencia/rango)
        df['Market_Phase'] = self._calculate_market_phase(df)
        
        # Distancia a máximos/mínimos de 52 semanas (252 días de trading)
        df['Distance_52W_High'] = self._calculate_distance_to_high(df['Close'], 252)
        df['Distance_52W_Low'] = self._calculate_distance_to_low(df['Close'], 252)
        
        # ADX Diario normalizado
        df['ADX'] = self._calculate_adx(df, 14)
        df['ADX_normalized'] = self._normalize(df['ADX'], 0, 100)
        
        return df
    
    def _calculate_hourly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para el timeframe horario"""
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        macd_df = self._calculate_macd(df['Close'], 12, 26, 9)
        df = pd.concat([df, macd_df], axis=1)
        
        # EMAs
        df['EMA8'] = self._calculate_ema(df['Close'], 8)
        df['EMA21'] = self._calculate_ema(df['Close'], 21)
        df['EMA50'] = self._calculate_ema(df['Close'], 50)
        
        # Bollinger Bands
        bb_df = self._calculate_bollinger_bands(df['Close'], 20, 2)
        df = pd.concat([df, bb_df], axis=1)
        
        # ATR
        df['ATR'] = self._calculate_atr(df, 14)
        
        # Stochastic Oscillator
        stoch_df = self._calculate_stochastic(df, 14, 3)
        df = pd.concat([df, stoch_df], axis=1)
        
        # OBV
        df['OBV'] = self._calculate_obv(df)
        
        # ADX
        df['ADX'] = self._calculate_adx(df, 14)
        
        return df
    
    def _calculate_15min_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para el timeframe de 15 minutos"""
        # Momentum de corto plazo
        df['Momentum'] = self._calculate_momentum(df['Close'], 5)
        
        # Divergencias RSI
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        df['RSI_divergence'] = self._calculate_rsi_divergence(df)
        
        # Aceleración de volumen
        df['Volume_acceleration'] = self._calculate_volume_acceleration(df['Volume'], 5)
        
        # Bollinger Squeeze
        df['BB_squeeze'] = self._calculate_bollinger_squeeze(df)
        
        return df
    
    def _calculate_cross_timeframe_features(self, data_1d: pd.DataFrame, data_1h: pd.DataFrame, 
                                           data_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características que combinan información de múltiples timeframes
        
        Args:
            data_1d: DataFrame con indicadores diarios
            data_1h: DataFrame con indicadores horarios
            data_15m: DataFrame con indicadores de 15 minutos
            
        Returns:
            DataFrame con características cross-timeframe
        """
        # Asegurarse de que los índices son datetime
        data_1d = data_1d.copy()
        data_1h = data_1h.copy()
        data_15m = data_15m.copy()
        
        # Reindexar para tener todos los timeframes en cada punto de 15 minutos
        result = pd.DataFrame(index=data_15m.index)
        
        # Alineación de tendencias
        result['trend_alignment'] = self._calculate_trend_alignment(data_1d, data_1h, data_15m)
        
        # Impulso multi-timeframe
        result['multi_tf_momentum'] = self._calculate_multi_tf_momentum(data_1d, data_1h, data_15m)
        
        # Detección de puntos de inflexión
        result['inflection_point'] = self._calculate_inflection_points(data_1d, data_1h, data_15m)
        
        # Índice de fuerza multi-timeframe
        result['multi_tf_strength'] = self._calculate_multi_tf_strength(data_1d, data_1h, data_15m)
        
        # Índice de cambio de tendencia (TCI)
        result['trend_change_index'] = self._calculate_trend_change_index(data_1d, data_1h, data_15m)
        
        # Ratio de volatilidad cross-timeframe
        result['volatility_ratio'] = self._calculate_volatility_ratio(data_1d, data_1h, data_15m)
        
        return result
    
    # Métodos para indicadores individuales usando implementaciones puras de Python
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula Relative Strength Index"""
        delta = series.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        
        roll_up = up.rolling(window=period).mean()
        roll_down = down.abs().rolling(window=period).mean()
        
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast_period: int, 
                       slow_period: int, signal_period: int) -> pd.DataFrame:
        """Calcula MACD (Moving Average Convergence Divergence)"""
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        macd_df = pd.DataFrame({
            'MACD': macd_line,
            'MACD_signal': signal_line,
            'MACD_histogram': macd_histogram
        }, index=series.index)
        
        return macd_df
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int, 
                                  std_dev: int) -> pd.DataFrame:
        """Calcula Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        bb_df = pd.DataFrame({
            'BB_middle': sma,
            'BB_upper': upper_band,
            'BB_lower': lower_band,
            'BB_width': (upper_band - lower_band) / sma,
            'BB_pct_b': (series - lower_band) / (upper_band - lower_band)
        }, index=series.index)
        
        return bb_df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcula Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int, 
                             d_period: int) -> pd.DataFrame:
        """Calcula Stochastic Oscillator"""
        high_roll = df['High'].rolling(window=k_period).max()
        low_roll = df['Low'].rolling(window=k_period).min()
        
        stoch_k = 100 * ((df['Close'] - low_roll) / (high_roll - low_roll))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        stoch_df = pd.DataFrame({
            'Stoch_K': stoch_k,
            'Stoch_D': stoch_d
        }, index=df.index)
        
        return stoch_df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calcula On-Balance Volume"""
        close = df['Close']
        volume = df['Volume']
        
        obv = pd.Series(0.0, index=close.index)
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcula Average Directional Index"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = pd.Series(0.0, index=up_move.index)
        neg_dm = pd.Series(0.0, index=down_move.index)
        
        pos_dm[((up_move > down_move) & (up_move > 0))] = up_move
        neg_dm[((down_move > up_move) & (down_move > 0))] = down_move
        
        # Directional Indicators
        # Evitar división por cero
        pdi = 100 * (pos_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
        ndi = 100 * (neg_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
        
        # Reemplazar NaN por 0
        pdi = pdi.fillna(0)
        ndi = ndi.fillna(0)
        
        # Directional Index
        dx_denom = (pdi + ndi).abs()
        dx = 100 * ((pdi - ndi).abs() / dx_denom.replace(0, np.nan)).fillna(0)
        
        # Average Directional Index
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula la pendiente de una serie"""
        slopes = pd.Series(index=series.index)
        
        for i in range(period, len(series)):
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            slope, _ = np.polyfit(x, y, 1)
            slopes.iloc[i] = slope
            
        return slopes
    
    def _calculate_market_phase(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula la fase del mercado:
        1: Tendencia alcista fuerte
        0.5: Tendencia alcista débil
        0: Rango
        -0.5: Tendencia bajista débil
        -1: Tendencia bajista fuerte
        """
        # Calcular ADX para medir fuerza de la tendencia
        adx = self._calculate_adx(df, 14)
        
        # Calcular dirección con EMA
        ema20 = self._calculate_ema(df['Close'], 20)
        ema50 = self._calculate_ema(df['Close'], 50)
        
        trend_direction = pd.Series(0.0, index=df.index)
        
        # Tendencia alcista: EMA20 > EMA50
        trend_direction[ema20 > ema50] = 1
        
        # Tendencia bajista: EMA20 < EMA50
        trend_direction[ema20 < ema50] = -1
        
        # Combinar dirección y fuerza de la tendencia
        market_phase = pd.Series(0.0, index=df.index)  # Valor por defecto: rango
        
        # Tendencia fuerte (ADX > 25)
        strong_trend = adx > 25
        market_phase[strong_trend & (trend_direction == 1)] = 1    # Alcista fuerte
        market_phase[strong_trend & (trend_direction == -1)] = -1  # Bajista fuerte
        
        # Tendencia débil (15 < ADX <= 25)
        weak_trend = (adx > 15) & (adx <= 25)
        market_phase[weak_trend & (trend_direction == 1)] = 0.5    # Alcista débil
        market_phase[weak_trend & (trend_direction == -1)] = -0.5  # Bajista débil
        
        return market_phase
    
    def _calculate_distance_to_high(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula la distancia porcentual al máximo del período"""
        rolling_max = series.rolling(window=period).max()
        distance = (series - rolling_max) / rolling_max
        return distance
    
    def _calculate_distance_to_low(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula la distancia porcentual al mínimo del período"""
        rolling_min = series.rolling(window=period).min()
        distance = (series - rolling_min) / rolling_min
        return distance
    
    def _normalize(self, series: pd.Series, min_val: float, max_val: float) -> pd.Series:
        """Normaliza una serie al rango [0, 1]"""
        return (series - min_val) / (max_val - min_val)
    
    def _calculate_momentum(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula el momentum de una serie"""
        return series / series.shift(period) - 1
    
    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula divergencias entre precio y RSI
        1: Divergencia alcista
        -1: Divergencia bajista
        0: Sin divergencia
        """
        price = df['Close']
        rsi = df['RSI']
        
        # Ventana para buscar máximos y mínimos locales
        window = 5
        
        divergence = pd.Series(0.0, index=price.index)
        
        for i in range(window, len(price) - window):
            # Comprobar si hay un mínimo local en el precio
            if (price.iloc[i] < price.iloc[i-window:i]).all() and (price.iloc[i] < price.iloc[i+1:i+window+1]).all():
                # Comprobar si el RSI está haciendo un mínimo más alto
                # (divergencia alcista)
                if rsi.iloc[i] > rsi.iloc[i-window:i].min():
                    divergence.iloc[i] = 1
                    
            # Comprobar si hay un máximo local en el precio
            if (price.iloc[i] > price.iloc[i-window:i]).all() and (price.iloc[i] > price.iloc[i+1:i+window+1]).all():
                # Comprobar si el RSI está haciendo un máximo más bajo
                # (divergencia bajista)
                if rsi.iloc[i] < rsi.iloc[i-window:i].max():
                    divergence.iloc[i] = -1
        
        return divergence
    
    def _calculate_volume_acceleration(self, volume: pd.Series, period: int) -> pd.Series:
        """Calcula la aceleración del volumen"""
        vol_mom = volume / volume.shift(1)
        vol_accel = vol_mom / vol_mom.shift(period)
        return vol_accel
    
    def _calculate_bollinger_squeeze(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta Bollinger Squeeze (bandas estrechas)
        1: Squeeze (bandas estrechas)
        0: No squeeze
        """
        # Necesitamos BB_width
        if 'BB_width' not in df.columns:
            bb_df = self._calculate_bollinger_bands(df['Close'], 20, 2)
            bb_width = bb_df['BB_width']
        else:
            bb_width = df['BB_width']
        
        # Calculamos un umbral para definir un squeeze
        # Típicamente se considera squeeze cuando el ancho está en el percentil 20 inferior
        threshold = bb_width.rolling(window=min(252, len(bb_width))).quantile(0.2)
        
        squeeze = pd.Series(0.0, index=df.index)
        squeeze[bb_width < threshold] = 1
        
        return squeeze
    
    # Métodos para características cross-timeframe
    
    def _resample_to_lower_timeframe(self, df: pd.DataFrame, 
                                    source_tf: str, target_tf: str) -> pd.DataFrame:
        """
        Resamplea un DataFrame de un timeframe superior a uno inferior
        
        Args:
            df: DataFrame con índice temporal
            source_tf: Timeframe de origen ('1d', '1h')
            target_tf: Timeframe de destino ('1h', '15m')
            
        Returns:
            DataFrame resampleado al timeframe inferior
        """
        result = pd.DataFrame(index=pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=self._get_freq(target_tf)
        ))
        
        # Para cada columna, aplicar forward fill
        for col in df.columns:
            # Primero reindexar con el nuevo índice, manteniendo solo los valores originales
            temp = df[col].reindex(result.index, method=None)
            
            # Luego aplicar forward fill para rellenar los espacios
            result[col] = temp.ffill()
            
        return result
    
    def _get_freq(self, timeframe: str) -> str:
        """Convierte el timeframe a la frecuencia de pandas"""
        if timeframe == '1d':
            return 'B'  # Días hábiles
        elif timeframe == '1h':
            return 'H'
        elif timeframe == '15m':
            return '15min'
        else:
            raise ValueError(f"Timeframe no soportado: {timeframe}")
    
    def _calculate_trend_alignment(self, data_1d: pd.DataFrame, 
                                  data_1h: pd.DataFrame, 
                                  data_15m: pd.DataFrame) -> pd.Series:
        """
        Calcula la alineación de tendencias entre timeframes
        
        Returns:
            Serie con valores entre -1 y 1:
            1: Todas las tendencias alcistas
            0.33: Tendencia alcista en 1 timeframe
            0: Tendencias mixtas
            -0.33: Tendencia bajista en 1 timeframe
            -1: Todas las tendencias bajistas
        """
        # Resamplear datos a 15m
        daily_15m = self._resample_to_lower_timeframe(data_1d, '1d', '15m')
        hourly_15m = self._resample_to_lower_timeframe(data_1h, '1h', '15m')
        
        # Verificar si tenemos las columnas necesarias
        if 'EMA8' not in daily_15m.columns or 'EMA21' not in daily_15m.columns:
            # Calcular si no existen
            if 'Close' in daily_15m.columns:
                daily_15m['EMA8'] = self._calculate_ema(daily_15m['Close'], 8)
                daily_15m['EMA21'] = self._calculate_ema(daily_15m['Close'], 21)
            else:
                # Si no tenemos datos de precio, usar tendencia neutral
                return pd.Series(0.0, index=data_15m.index)
                
        if 'EMA8' not in hourly_15m.columns or 'EMA21' not in hourly_15m.columns:
            if 'Close' in hourly_15m.columns:
                hourly_15m['EMA8'] = self._calculate_ema(hourly_15m['Close'], 8)
                hourly_15m['EMA21'] = self._calculate_ema(hourly_15m['Close'], 21)
            else:
                return pd.Series(0.0, index=data_15m.index)
                
        if 'EMA8' not in data_15m.columns or 'EMA21' not in data_15m.columns:
            if 'Close' in data_15m.columns:
                data_15m['EMA8'] = self._calculate_ema(data_15m['Close'], 8)
                data_15m['EMA21'] = self._calculate_ema(data_15m['Close'], 21)
            else:
                return pd.Series(0.0, index=data_15m.index)
            
        # Obtener direcciones de tendencia
        trend_d = daily_15m['EMA8'] > daily_15m['EMA21']
        trend_h = hourly_15m['EMA8'] > hourly_15m['EMA21']
        trend_m = data_15m['EMA8'] > data_15m['EMA21']
        
        # Convertir booleanos a 1 (alcista) y -1 (bajista)
        trend_d = trend_d.astype(int) * 2 - 1
        trend_h = trend_h.astype(int) * 2 - 1
        trend_m = trend_m.astype(int) * 2 - 1
        
        # Calcular alineación como promedio
        alignment = (trend_d + trend_h + trend_m) / 3
        
        return alignment
    
    def _calculate_multi_tf_momentum(self, data_1d: pd.DataFrame, 
                                    data_1h: pd.DataFrame, 
                                    data_15m: pd.DataFrame) -> pd.Series:
        """
        Calcula el impulso combinado de múltiples timeframes
        """
        # Resamplear datos a 15m
        daily_15m = self._resample_to_lower_timeframe(data_1d, '1d', '15m')
        hourly_15m = self._resample_to_lower_timeframe(data_1h, '1h', '15m')
        
        # Verificar si tenemos las columnas necesarias
        mom_d = pd.Series(0.0, index=data_15m.index)
        mom_h = pd.Series(0.0, index=data_15m.index)
        mom_m = pd.Series(0.0, index=data_15m.index)
        
        # Calcular momentum en cada timeframe (como % de cambio)
        if 'Momentum' in daily_15m.columns:
            mom_d = daily_15m['Momentum'].fillna(0)
        elif 'Close' in daily_15m.columns:
            mom_d = self._calculate_momentum(daily_15m['Close'], 5).fillna(0)
            
        if 'Momentum' in hourly_15m.columns:
            mom_h = hourly_15m['Momentum'].fillna(0)
        elif 'Close' in hourly_15m.columns:
            mom_h = self._calculate_momentum(hourly_15m['Close'], 5).fillna(0)
            
        if 'Momentum' in data_15m.columns:
            mom_m = data_15m['Momentum'].fillna(0)
        elif 'Close' in data_15m.columns:
            mom_m = self._calculate_momentum(data_15m['Close'], 5).fillna(0)
        
        # Normalizar cada momentum
        mom_d = self._normalize_momentum(mom_d)
        mom_h = self._normalize_momentum(mom_h)
        mom_m = self._normalize_momentum(mom_m)
        
        # Combinar con pesos (dando más importancia a timeframes mayores)
        multi_mom = (mom_d * 0.5) + (mom_h * 0.3) + (mom_m * 0.2)
        
        return multi_mom
    
    def _normalize_momentum(self, mom: pd.Series) -> pd.Series:
        """Normaliza el momentum a un rango [-1, 1]"""
        # Calcular percentiles para identificar valores extremos
        window_size = min(100, len(mom))
        if window_size < 5:  # Si hay muy pocos datos, devolver los valores originales recortados
            return mom.clip(-1, 1)
            
        p_high = mom.rolling(window=window_size).quantile(0.95)
        p_low = mom.rolling(window=window_size).quantile(0.05)
        
        # Normalizar
        norm_mom = pd.Series(index=mom.index)
        
        # Valores positivos
        mask_pos = mom > 0
        norm_mom[mask_pos] = mom[mask_pos] / p_high[mask_pos].replace(0, np.nan)
        
        # Valores negativos
        mask_neg = mom < 0
        norm_mom[mask_neg] = mom[mask_neg] / abs(p_low[mask_neg].replace(0, np.nan))
        
        # Reemplazar NaN por 0
        norm_mom = norm_mom.fillna(0)
        
        # Limitar a [-1, 1]
        norm_mom = norm_mom.clip(-1, 1)
        
        return norm_mom
    
    def _calculate_inflection_points(self, data_1d: pd.DataFrame, 
                                    data_1h: pd.DataFrame, 
                                    data_15m: pd.DataFrame) -> pd.Series:
        """
        Detecta posibles puntos de inflexión basados en el análisis multi-timeframe
        
        Returns:
            Serie con valores:
            1: Posible punto de inflexión alcista
            -1: Posible punto de inflexión bajista
            0: No es punto de inflexión
        """
        # Resamplear datos a 15m
        daily_15m = self._resample_to_lower_timeframe(data_1d, '1d', '15m')
        hourly_15m = self._resample_to_lower_timeframe(data_1h, '1h', '15m')
        
        # Inicializar serie de resultado
        inflection = pd.Series(0.0, index=data_15m.index)
        
        # Verificar si tenemos todas las columnas necesarias
        if 'RSI' not in data_15m.columns or 'RSI' not in hourly_15m.columns or 'Market_Phase' not in daily_15m.columns:
            return inflection
            
        if 'RSI_divergence' not in data_15m.columns:
            # Si no tenemos divergencia calculada, usar un valor por defecto
            data_15m['RSI_divergence'] = 0
        
        # Condiciones para punto de inflexión alcista
        long_cond = (
            # RSI en sobreventa en timeframes inferiores
            (data_15m['RSI'] < 30) & 
            (hourly_15m['RSI'] < 40) &
            # Divergencia alcista en 15m
            (data_15m['RSI_divergence'] == 1) &
            # Tendencia diaria no fuertemente bajista
            (daily_15m['Market_Phase'] >= -0.5)
        )
        
        # Condiciones para punto de inflexión bajista
        short_cond = (
            # RSI en sobrecompra en timeframes inferiores
            (data_15m['RSI'] > 70) & 
            (hourly_15m['RSI'] > 60) &
            # Divergencia bajista en 15m
            (data_15m['RSI_divergence'] == -1) &
            # Tendencia diaria no fuertemente alcista
            (daily_15m['Market_Phase'] <= 0.5)
        )
        
        # Asignar valores
        inflection[long_cond] = 1
        inflection[short_cond] = -1
        
        return inflection
    
    def _calculate_multi_tf_strength(self, data_1d: pd.DataFrame, 
                                   data_1h: pd.DataFrame, 
                                   data_15m: pd.DataFrame) -> pd.Series:
        """
        Calcula un índice de fuerza multi-timeframe
        """
        # Resamplear datos a 15m
        daily_15m = self._resample_to_lower_timeframe(data_1d, '1d', '15m')
        hourly_15m = self._resample_to_lower_timeframe(data_1h, '1h', '15m')
        
        # Índice de fuerza por defecto
        strength = pd.Series(0.0, index=data_15m.index)
        
        # Verificar si tenemos todas las columnas necesarias
        if 'RSI' not in daily_15m.columns or 'RSI' not in hourly_15m.columns or 'RSI' not in data_15m.columns:
            # Calcular RSI si tenemos datos de precio
            if 'Close' in daily_15m.columns:
                daily_15m['RSI'] = self._calculate_rsi(daily_15m['Close'], 14)
            if 'Close' in hourly_15m.columns:
                hourly_15m['RSI'] = self._calculate_rsi(hourly_15m['Close'], 14)
            if 'Close' in data_15m.columns:
                data_15m['RSI'] = self._calculate_rsi(data_15m['Close'], 14)
                
        if 'Market_Phase' not in daily_15m.columns:
            if 'Close' in daily_15m.columns and 'High' in daily_15m.columns and 'Low' in daily_15m.columns:
                daily_15m['Market_Phase'] = self._calculate_market_phase(daily_15m)
                
        if 'MACD_histogram' not in hourly_15m.columns:
            if 'Close' in hourly_15m.columns:
                macd_df = self._calculate_macd(hourly_15m['Close'], 12, 26, 9)
                hourly_15m['MACD_histogram'] = macd_df['MACD_histogram']
                
        if 'Momentum' not in data_15m.columns:
            if 'Close' in data_15m.columns:
                data_15m['Momentum'] = self._calculate_momentum(data_15m['Close'], 5)
        
        # Verificar nuevamente si tenemos todas las columnas necesarias
        required_cols = {
            'daily': ['RSI', 'Market_Phase'],
            'hourly': ['RSI', 'MACD_histogram'],
            'min15': ['RSI', 'Momentum']
        }
        
        missing_cols = False
        for df_name, cols in [('daily_15m', required_cols['daily']), 
                             ('hourly_15m', required_cols['hourly']), 
                             ('data_15m', required_cols['min15'])]:
            df = locals()[df_name]
            for col in cols:
                if col not in df.columns:
                    missing_cols = True
                    break
            if missing_cols:
                break
                
        if missing_cols:
            return strength
            
        # Fuerza diaria (normalizada a [-1, 1])
        d_rsi = (daily_15m['RSI'] - 50) / 50
        d_trend = (daily_15m['Market_Phase']).clip(-1, 1)
        
        # Fuerza horaria
        h_rsi = (hourly_15m['RSI'] - 50) / 50
        h_macd = self._normalize_macd(hourly_15m['MACD_histogram'])
        
        # Fuerza 15 minutos
        m_rsi = (data_15m['RSI'] - 50) / 50
        m_momentum = data_15m['Momentum'].clip(-1, 1)
        
        # Combinar con pesos
        strength = (
            (d_rsi * 0.2 + d_trend * 0.3) +  # 50% diario
            (h_rsi * 0.15 + h_macd * 0.15) +  # 30% horario
            (m_rsi * 0.1 + m_momentum * 0.1)   # 20% 15 minutos
        )
        
        return strength
    
    def _normalize_macd(self, macd_hist: pd.Series) -> pd.Series:
        """Normaliza el histograma MACD a rango [-1, 1]"""
        # Calcular percentiles para identificar valores extremos
        window_size = min(100, len(macd_hist))
        if window_size < 5:  # Si hay muy pocos datos, devolver los valores originales recortados
            return macd_hist.clip(-1, 1)
            
        p_high = macd_hist.rolling(window=window_size).quantile(0.95)
        p_low = macd_hist.rolling(window=window_size).quantile(0.05)
        
        # Normalizar
        norm_macd = pd.Series(index=macd_hist.index)
        
        # Valores positivos
        mask_pos = macd_hist > 0
        norm_macd[mask_pos] = macd_hist[mask_pos] / p_high[mask_pos].replace(0, np.nan)
        
        # Valores negativos
        mask_neg = macd_hist < 0
        norm_macd[mask_neg] = macd_hist[mask_neg] / abs(p_low[mask_neg].replace(0, np.nan))
        
        # Reemplazar NaN por 0
        norm_macd = norm_macd.fillna(0)
        
        # Limitar a [-1, 1]
        norm_macd = norm_macd.clip(-1, 1)
        
        return norm_macd
    
    def _calculate_trend_change_index(self, data_1d: pd.DataFrame, 
                                data_1h: pd.DataFrame, 
                                data_15m: pd.DataFrame) -> pd.Series:
        """
        Calcula un índice de probabilidad de cambio de tendencia
        
        Returns:
            Serie con valores de 0 a 1:
            0: Baja probabilidad de cambio de tendencia
            1: Alta probabilidad de cambio de tendencia
        """
        # Verificar si algún DataFrame está vacío
        if data_1d.empty or data_1h.empty or data_15m.empty:
            print("Warning: Al menos uno de los DataFrames está vacío en _calculate_trend_change_index")
            return pd.Series(0.0, index=data_15m.index if not data_15m.empty else pd.DatetimeIndex([]))
        
        try:
            # Resamplear datos a 15m con manejo de errores
            try:
                daily_15m = self._resample_to_lower_timeframe(data_1d, '1d', '15m')
            except Exception as e:
                print(f"Error al resamplear data_1d: {e}")
                daily_15m = pd.DataFrame(index=data_15m.index)
                
            try:
                hourly_15m = self._resample_to_lower_timeframe(data_1h, '1h', '15m')
            except Exception as e:
                print(f"Error al resamplear data_1h: {e}")
                hourly_15m = pd.DataFrame(index=data_15m.index)
            
            # Inicializar serie de resultado con ceros
            tci = pd.Series(0.0, index=data_15m.index)
            
            # Verificar y calcular características necesarias
            
            # 1. Divergencias en RSI
            has_divergence = pd.Series(0.0, index=data_15m.index)
            if 'RSI_divergence' in data_15m.columns:
                has_divergence = data_15m['RSI_divergence'].abs() > 0
                has_divergence = has_divergence.astype(float)  # Convertir a float
            
            # 2. Squeeze de Bollinger y posterior expansión
            squeeze_expanding = pd.Series(0.0, index=data_15m.index)
            if 'BB_squeeze' in data_15m.columns:
                # Verificar si BB_squeeze tiene algún valor
                if not data_15m['BB_squeeze'].isna().all():
                    # CORRECCIÓN: Asegurarse de usar comparaciones booleanas correctamente
                    # Un squeeze seguido de expansión: BB_squeeze cambia de 1 a 0
                    try:
                        # Calcular donde hay 1 seguido de 0 (squeeze que termina)
                        squeeze_end = (data_15m['BB_squeeze'].shift(1) == 1) & (data_15m['BB_squeeze'] == 0)
                        squeeze_expanding = squeeze_end.astype(float)  # Convertir a float
                    except Exception as e:
                        print(f"Error al calcular squeeze_expanding: {e}")
                        # Proporcionar más información de diagnóstico
                        print(f"Tipos de datos: BB_squeeze={data_15m['BB_squeeze'].dtype}")
                        print(f"Valores únicos en BB_squeeze: {data_15m['BB_squeeze'].unique()}")
                        # En caso de error, mantener valor por defecto
            
            # 3. Impulso fuerte en el menor timeframe contra la tendencia principal
            counter_momentum = pd.Series(0.0, index=data_15m.index)
            if 'Market_Phase' in daily_15m.columns and 'Momentum' in data_15m.columns:
                try:
                    # Convertir a valores numéricos si es necesario
                    if not pd.api.types.is_numeric_dtype(daily_15m['Market_Phase']):
                        daily_15m['Market_Phase'] = pd.to_numeric(daily_15m['Market_Phase'], errors='coerce').fillna(0)
                    if not pd.api.types.is_numeric_dtype(data_15m['Momentum']):
                        data_15m['Momentum'] = pd.to_numeric(data_15m['Momentum'], errors='coerce').fillna(0)
                        
                    # Tendencia diaria alcista pero impulso 15m bajista fuerte
                    cond1 = (daily_15m['Market_Phase'] > 0) & (data_15m['Momentum'] < -0.01)
                    # Tendencia diaria bajista pero impulso 15m alcista fuerte
                    cond2 = (daily_15m['Market_Phase'] < 0) & (data_15m['Momentum'] > 0.01)
                    
                    counter_momentum = (cond1 | cond2).astype(float)
                except Exception as e:
                    print(f"Error al calcular counter_momentum: {e}")
            
            # 4. Cruce de medias móviles en timeframe inferior
            ema_cross = pd.Series(0.0, index=data_15m.index)
            if 'EMA8' in data_15m.columns and 'EMA21' in data_15m.columns:
                try:
                    # Cruce hacia arriba: EMA8 cruza por encima de EMA21
                    up_cross = (data_15m['EMA8'].shift(1) < data_15m['EMA21'].shift(1)) & (data_15m['EMA8'] > data_15m['EMA21'])
                    # Cruce hacia abajo: EMA8 cruza por debajo de EMA21
                    down_cross = (data_15m['EMA8'].shift(1) > data_15m['EMA21'].shift(1)) & (data_15m['EMA8'] < data_15m['EMA21'])
                    
                    ema_cross = (up_cross | down_cross).astype(float)
                except Exception as e:
                    print(f"Error al calcular ema_cross: {e}")
            
            # 5. Volumen inusual
            high_volume = pd.Series(0.0, index=data_15m.index)
            if 'Volume_acceleration' in data_15m.columns:
                try:
                    if not pd.api.types.is_numeric_dtype(data_15m['Volume_acceleration']):
                        data_15m['Volume_acceleration'] = pd.to_numeric(data_15m['Volume_acceleration'], errors='coerce').fillna(1.0)
                    high_volume = (data_15m['Volume_acceleration'] > 1.5).astype(float)
                except Exception as e:
                    print(f"Error al calcular high_volume: {e}")
            
            # Combinar factores con pesos
            try:
                tci = (
                    has_divergence * 0.25 +
                    squeeze_expanding * 0.2 +
                    counter_momentum * 0.2 +
                    ema_cross * 0.15 +
                    high_volume * 0.2
                )
                
                # Asegurarse de que el resultado está en el rango [0, 1]
                tci = tci.clip(0, 1)
            except Exception as e:
                print(f"Error al combinar factores para TCI: {e}")
                # En caso de error, devolver serie de ceros
                tci = pd.Series(0.0, index=data_15m.index)
            
            return tci
            
        except Exception as e:
            print(f"Error general en _calculate_trend_change_index: {e}")
            # Devolver serie de ceros en caso de error
            return pd.Series(0.0, index=data_15m.index)

    def _calculate_volatility_ratio(self, data_1d: pd.DataFrame, 
                                  data_1h: pd.DataFrame, 
                                  data_15m: pd.DataFrame) -> pd.Series:
        """
        Calcula el ratio de volatilidad entre timeframes
        
        Un valor alto indica que la volatilidad en timeframes inferiores
        es alta en relación a timeframes superiores, lo que puede anticipar
        un posible cambio de tendencia
        """
        # Resamplear datos a 15m
        daily_15m = self._resample_to_lower_timeframe(data_1d, '1d', '15m')
        hourly_15m = self._resample_to_lower_timeframe(data_1h, '1h', '15m')
        
        # Ratio por defecto
        vol_ratio_norm = pd.Series(0.5, index=data_15m.index)
        
        # Verificar y calcular ATR si no existe
        dfs = [daily_15m, hourly_15m, data_15m]
        for i, df in enumerate(dfs):
            if 'ATR' not in df.columns:
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    df['ATR'] = self._calculate_atr(df, 14)
        
        # Verificar nuevamente
        if not all('ATR' in df.columns for df in dfs):
            return vol_ratio_norm
        
        # Calcular ratios
        price = data_15m['Close']
        
        daily_atr_norm = daily_15m['ATR'] / price
        hourly_atr_norm = hourly_15m['ATR'] / price
        min15_atr_norm = data_15m['ATR'] / price
        
        # Calcular ratios
        ratio_m_to_h = min15_atr_norm / hourly_atr_norm.replace(0, np.nan)
        ratio_h_to_d = hourly_atr_norm / daily_atr_norm.replace(0, np.nan)
        
        # Reemplazar NaN por 1 (ratio neutro)
        ratio_m_to_h = ratio_m_to_h.fillna(1)
        ratio_h_to_d = ratio_h_to_d.fillna(1)
        
        # Combinar en un indicador
        volatility_ratio = (ratio_m_to_h * 0.7 + ratio_h_to_d * 0.3)
        
        # Normalizar a [0, 1] basado en percentiles históricos
        window_size = min(100, len(volatility_ratio))
        if window_size < 5:
            return vol_ratio_norm
            
        min_val = volatility_ratio.rolling(window_size).min()
        max_val = volatility_ratio.rolling(window_size).max()
        
        denom = (max_val - min_val).replace(0, np.nan)
        vol_ratio_norm = (volatility_ratio - min_val) / denom
        
        # Reemplazar NaN por 0.5 (valor neutral)
        vol_ratio_norm = vol_ratio_norm.fillna(0.5)
        
        return vol_ratio_norm.clip(0, 1)

# # Ejemplo de uso con datos de pandas
# if __name__ == "__main__":
#     # Supongamos que ya tenemos datos cargados
#     # Puedes usar yfinance para pruebas
#     import yfinance as yf
    
#     # Descargar datos
#     ticker = "AAPL"
#     data_1d = yf.download(ticker, period="1y", interval="1d")
#     data_1h = yf.download(ticker, period="60d", interval="1h")
#     data_15m = yf.download(ticker, period="7d", interval="15m")
    
#     # Crear instancia de la clase
#     feature_eng = FeatureEngineering()
    
#     # Calcular indicadores para cada timeframe
#     features_1d = feature_eng.calculate_indicators(data_1d, '1d')
#     features_1h = feature_eng.calculate_indicators(data_1h, '1h')
#     features_15m = feature_eng.calculate_indicators(data_15m, '15m')
    
#     # Calcular características cross-timeframe
#     cross_tf_features = feature_eng._calculate_cross_timeframe_features(
#         features_1d, features_1h, features_15m
#     )
    
#     print(f"Características diarias: {features_1d.columns.tolist()}")
#     print(f"Características horarias: {features_1h.columns.tolist()}")
#     print(f"Características 15 min: {features_15m.columns.tolist()}")
#     print(f"Características cross-timeframe: {cross_tf_features.columns.tolist()}")