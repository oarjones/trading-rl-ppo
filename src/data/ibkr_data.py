import pandas as pd
import numpy as np
import datetime
import time
import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from ib_insync import IB, Contract, util, Stock, BarData
import asyncio

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IBDataManager:
    """
    Gestiona la obtención y almacenamiento de datos históricos de IBKR
    usando ib_insync, una biblioteca de alto nivel para la API de Interactive Brokers
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1,
                 timeout: int = 30):
        """
        Inicializa el gestor de datos de IBKR
        
        Args:
            host: Host de IBKR TWS o Gateway (por defecto localhost)
            port: Puerto (7497 para TWS paper, 7496 para TWS real, 4002 para Gateway)
            client_id: ID de cliente para la conexión
            timeout: Tiempo de espera para operaciones en segundos
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.ib = IB()  # Inicializar objeto IB
        self.connected = False
        self.data_cache = {}
        
    def connect(self) -> bool:
        """Establece conexión con IBKR TWS/Gateway"""
        try:
            if not self.connected:
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
                self.connected = self.ib.isConnected()
                if self.connected:
                    logger.info(f"Conectado a IBKR en {self.host}:{self.port}")
                else:
                    logger.error("No se pudo establecer conexión con IBKR")
            return self.connected
        except Exception as e:
            logger.error(f"Error al conectar con IBKR: {e}")
            return False
            
    def disconnect(self):
        """Cierra la conexión con IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Desconectado de IBKR")
            
    def create_contract(self, symbol: str, sec_type: str = "STK", 
                      exchange: str = "SMART", currency: str = "USD", 
                      **kwargs) -> Contract:
        """
        Crea un objeto Contract para usar con la API de IBKR
        
        Args:
            symbol: Símbolo del instrumento
            sec_type: Tipo de seguridad (STK, FUT, OPT, CASH, etc.)
            exchange: Mercado (SMART, ISLAND, NYSE, etc.)
            currency: Moneda (USD, EUR, etc.)
            **kwargs: Argumentos adicionales para tipos específicos de contratos
            
        Returns:
            Objeto Contract configurado
        """
        if sec_type == "STK":
            # Crear contrato de acciones
            contract = Stock(symbol, exchange, currency)
        else:
            # Crear contrato genérico para otros tipos
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency
            
            # Agregar argumentos adicionales si son proporcionados
            for key, value in kwargs.items():
                if hasattr(contract, key):
                    setattr(contract, key, value)
                    
        return contract
    
    def get_contract_details(self, symbol: str, sec_type: str = "STK", 
                          exchange: str = "SMART", currency: str = "USD", 
                          **kwargs) -> List:
        """
        Obtiene los detalles de un contrato
        
        Args:
            Mismos argumentos que create_contract
            
        Returns:
            Lista de detalles de contrato
        """
        if not self.connected and not self.connect():
            logger.error("No se pudo conectar a IBKR")
            return []
            
        try:
            # Crear contrato
            contract = self.create_contract(symbol, sec_type, exchange, currency, **kwargs)
            
            # Obtener detalles del contrato
            details = self.ib.reqContractDetails(contract)
            
            logger.info(f"Obtenidos {len(details)} detalles para {symbol}")
            return details
        except Exception as e:
            logger.error(f"Error al obtener detalles del contrato para {symbol}: {e}")
            return []
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                     duration: str, sec_type: str = "STK", 
                     exchange: str = "SMART", currency: str = "USD",
                     use_rth: bool = False, **kwargs) -> pd.DataFrame:
        """
        Obtiene datos históricos para un símbolo y timeframe
        
        Args:
            symbol: Símbolo del instrumento
            timeframe: Timeframe de los datos ('1D', '1H', '15M', etc.)
            duration: Duración de los datos ('1 Y', '6 M', '1 W', etc.)
            sec_type, exchange, currency: Parámetros del contrato
            use_rth: Si es True, solo usa datos dentro del horario regular de trading
            **kwargs: Argumentos adicionales para el contrato
            
        Returns:
            DataFrame con los datos históricos
        """
        if not self.connected and not self.connect():
            logger.error("No se pudo conectar a IBKR")
            return pd.DataFrame()
            
        # Mapear timeframe al formato de IBKR
        bar_size_map = {
            '1D': '1 day',
            '1H': '1 hour',
            '15M': '15 mins',
            '5M': '5 mins',
            '1M': '1 min'
        }
        
        if timeframe not in bar_size_map:
            logger.error(f"Timeframe no soportado: {timeframe}")
            return pd.DataFrame()
            
        bar_size = bar_size_map[timeframe]
        
        try:
            # Crear contrato
            contract = self.create_contract(symbol, sec_type, exchange, currency, **kwargs)
            
            # Obtener datos históricos
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime='',  # '' para la fecha actual
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=use_rth,
                formatDate=1  # 1 = formato de fecha como cadena "YYYYMMDD"
            )
            
            if not bars:
                logger.warning(f"No se obtuvieron datos para {symbol} en timeframe {timeframe}")
                return pd.DataFrame()
                
            # Convertir a DataFrame
            df = util.df(bars)
            
            # Verificar si el DataFrame está vacío
            if df.empty:
                logger.warning(f"DataFrame vacío para {symbol} en timeframe {timeframe}")
                return df
                
            # Renombrar columnas para consistencia
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Verificar si las columnas existen antes de renombrarlas
            for old_col, new_col in column_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
                    
            # Establecer fecha como índice si no lo está ya
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
                
            # Asegurarse de que el índice es datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            
                
            # Almacenar en caché para futuras consultas
            cache_key = f"{symbol}_{timeframe}"
            self.data_cache[cache_key] = df.copy()
            
            logger.info(f"Obtenidos {len(df)} registros para {symbol} en timeframe {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos para {symbol} en timeframe {timeframe}: {e}")
            return pd.DataFrame()

    def get_data_for_multiple_assets(self, symbols: List[str], timeframes: List[str], 
                                   duration: str, use_rth: bool = True,
                                   sec_type: str = "STK", exchange: str = "SMART", 
                                   currency: str = "USD") -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Obtiene datos históricos para múltiples activos y timeframes
        
        Args:
            symbols: Lista de símbolos
            timeframes: Lista de timeframes ('1D', '1H', '15M', etc.)
            duration: Duración de los datos ('1 Y', '6 M', '1 W', etc.)
            use_rth: Si es True, solo usa datos dentro del horario regular de trading
            sec_type, exchange, currency: Parámetros del contrato
            
        Returns:
            Diccionario de diccionarios: {símbolo: {timeframe: DataFrame}}
        """
        result = {}
        
        if not self.connected and not self.connect():
            logger.error("No se pudo conectar a IBKR")
            return result
            
        for symbol in symbols:
            result[symbol] = {}
            
            for tf in timeframes:
                logger.info(f"Obteniendo datos para {symbol} en timeframe {tf}")
                
                # Obtener datos
                df = self.get_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    duration=duration,
                    sec_type=sec_type,
                    exchange=exchange,
                    currency=currency,
                    use_rth=use_rth
                )
                
                if not df.empty:
                    result[symbol][tf] = df
                    
                # Esperar un poco para evitar throttling de IBKR
                time.sleep(0.5)
                
        return result
    
    def save_data_to_disk(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                        output_dir: str = "data"):
        """
        Guarda los datos descargados en disco
        
        Args:
            data: Datos en formato {símbolo: {timeframe: DataFrame}}
            output_dir: Directorio de salida
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, timeframes_data in data.items():
            # Crear subdirectorio para el símbolo
            symbol_dir = os.path.join(output_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            for timeframe, df in timeframes_data.items():
                # Guardar como CSV
                file_path = os.path.join(symbol_dir, f"{timeframe}.csv")
                df.to_csv(file_path)
                logger.info(f"Datos guardados en {file_path}")
                
    def load_data_from_disk(self, symbols: List[str], timeframes: List[str], 
                      data_dir: str = "data") -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Carga datos desde disco
        
        Args:
            symbols: Lista de símbolos
            timeframes: Lista de timeframes
            data_dir: Directorio donde están almacenados los datos
            
        Returns:
            Datos en formato {símbolo: {timeframe: DataFrame}}
        """
        result = {}
        
        for symbol in symbols:
            result[symbol] = {}
            symbol_dir = os.path.join(data_dir, symbol)
            
            if not os.path.exists(symbol_dir):
                logger.warning(f"No existen datos para {symbol} en {data_dir}")
                continue
                
            for tf in timeframes:
                file_path = os.path.join(symbol_dir, f"{tf}.csv")
                
                if os.path.exists(file_path):
                    # Cargar datos con parse_dates=True para convertir el índice a datetime
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    if hasattr(df.index, 'tz'):
                        if df.index.tz is None:
                            df.index = df.index.tz_localize('UTC')
                        else:
                            df.index = df.index.tz_convert('UTC')
                        
                    result[symbol][tf] = df
                    logger.info(f"Datos cargados desde {file_path}")
                else:
                    logger.warning(f"No existen datos para {symbol} en timeframe {tf}")
                    
        return result

    def align_timeframes(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Alinea todos los timeframes para cada símbolo, asegurando consistencia temporal
        
        Args:
            data: Datos en formato {símbolo: {timeframe: DataFrame}}
            
        Returns:
            Datos alineados en el mismo formato
        """
        aligned_data = {}
        
        for symbol, timeframes_data in data.items():
            aligned_data[symbol] = {}
            
            # Verificar si tenemos todos los timeframes necesarios
            if not all(tf in timeframes_data for tf in ['1D', '1H', '15M']):
                logger.warning(f"Faltan timeframes para {symbol}, saltando alineación")
                aligned_data[symbol] = timeframes_data.copy()
                continue
                
            # Obtener dataframes
            df_1d = timeframes_data['1D'].copy()
            df_1h = timeframes_data['1H'].copy()
            df_15m = timeframes_data['15M'].copy()
            

            # Convertir todos a UTC
            if hasattr(df_1d.index, 'tz'):
                if df_1d.index.tz is None:
                    df_1d.index = df_1d.index.tz_localize('UTC')
                else:
                    df_1d.index = df_1d.index.tz_convert('UTC')


            # Convertir todos a UTC
            if hasattr(df_1h.index, 'tz'):
                if df_1h.index.tz is None:
                    df_1h.index = df_1h.index.tz_localize('UTC')
                else:
                    df_1h.index = df_1h.index.tz_convert('UTC')

            # Convertir todos a UTC
            if hasattr(df_15m.index, 'tz'):
                if df_15m.index.tz is None:
                    df_15m.index = df_15m.index.tz_localize('UTC')
                else:
                    df_15m.index = df_15m.index.tz_convert('UTC')
            
            # Encontrar el rango de fechas común
            start_date = max(df_1d.index.min(), df_1h.index.min(), df_15m.index.min())
            end_date = min(df_1d.index.max(), df_1h.index.max(), df_15m.index.max())
            
            # Recortar datos al rango común
            df_1d = df_1d[(df_1d.index >= start_date) & (df_1d.index <= end_date)]
            df_1h = df_1h[(df_1h.index >= start_date) & (df_1h.index <= end_date)]
            df_15m = df_15m[(df_15m.index >= start_date) & (df_15m.index <= end_date)]
            
            # Guardar datos alineados
            aligned_data[symbol]['1D'] = df_1d
            aligned_data[symbol]['1H'] = df_1h
            aligned_data[symbol]['15M'] = df_15m
            
            logger.info(f"Datos alineados para {symbol} desde {start_date} hasta {end_date}")
            
        return aligned_data
    
    async def async_get_historical_data(self, symbol: str, timeframe: str, 
                                      duration: str, sec_type: str = "STK", 
                                      exchange: str = "SMART", currency: str = "USD",
                                      use_rth: bool = False, **kwargs) -> pd.DataFrame:
        """
        Versión asíncrona para obtener datos históricos
        
        Útil para descargar múltiples conjuntos de datos simultáneamente
        """
        # Mapear timeframe al formato de IBKR
        bar_size_map = {
            '1D': '1 day',
            '1H': '1 hour',
            '15M': '15 mins',
            '5M': '5 mins',
            '1M': '1 min'
        }
        
        if timeframe not in bar_size_map:
            logger.error(f"Timeframe no soportado: {timeframe}")
            return pd.DataFrame()
            
        bar_size = bar_size_map[timeframe]
        
        try:
            # Crear contrato
            contract = self.create_contract(symbol, sec_type, exchange, currency, **kwargs)
            
            # Obtener datos históricos
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=use_rth,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No se obtuvieron datos para {symbol} en timeframe {timeframe}")
                return pd.DataFrame()
                
            # Convertir a DataFrame
            df = util.df(bars)
            
            # Renombrar columnas para consistencia
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            for old_col, new_col in column_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
                    
            # Establecer fecha como índice
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
                
            # Asegurarse de que el índice es datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            
                
            logger.info(f"Obtenidos {len(df)} registros para {symbol} en timeframe {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos para {symbol} en timeframe {timeframe}: {e}")
            return pd.DataFrame()
            
    async def async_get_data_for_multiple_assets(self, symbols: List[str], timeframes: List[str], 
                                              duration: str, use_rth: bool = True,
                                              sec_type: str = "STK", exchange: str = "SMART", 
                                              currency: str = "USD") -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Versión asíncrona para obtener datos de múltiples activos y timeframes simultáneamente
        """
        if not self.connected and not self.connect():
            logger.error("No se pudo conectar a IBKR")
            return {}
            
        result = {}
        tasks = []
        
        # Crear todas las tareas
        for symbol in symbols:
            result[symbol] = {}
            
            for tf in timeframes:
                # Crear tarea asíncrona para cada símbolo y timeframe
                task = self.async_get_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    duration=duration,
                    sec_type=sec_type,
                    exchange=exchange,
                    currency=currency,
                    use_rth=use_rth
                )
                
                tasks.append((symbol, tf, task))
                
        # Ejecutar todas las tareas concurrentemente
        for symbol, tf, task in tasks:
            df = await task
            if not df.empty:
                result[symbol][tf] = df
                
        return result

# Ejemplo de uso:
if __name__ == "__main__":
    # Inicializar gestor de datos
    data_manager = IBDataManager(port=7497)  # 7497 para TWS en modo paper
    
    try:
        # Conectar con IBKR
        if data_manager.connect():
            # Definir símbolos y timeframes
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            timeframes = ["1D", "1H", "15M"]
            
            # Obtener datos - versión sincrónica
            # data = data_manager.get_data_for_multiple_assets(
            #     symbols=symbols,
            #     timeframes=timeframes,
            #     duration="1 Y",  # 1 año de datos
            #     use_rth=True     # Solo horario regular de trading
            # )
            
            # Obtener datos - versión asíncrona (más rápida)
            async def main():
                data = await data_manager.async_get_data_for_multiple_assets(
                    symbols=symbols,
                    timeframes=timeframes,
                    duration="1 Y",
                    use_rth=True
                )
                
                # Alinear datos
                aligned_data = data_manager.align_timeframes(data)
                
                # Guardar datos
                data_manager.save_data_to_disk(aligned_data, output_dir="data")
                
                # Ejemplo de acceso a los datos
                for symbol in symbols:
                    if symbol in aligned_data:
                        for tf in timeframes:
                            if tf in aligned_data[symbol]:
                                df = aligned_data[symbol][tf]
                                print(f"{symbol} - {tf}: {len(df)} registros desde {df.index.min()} hasta {df.index.max()}")
            
            # Ejecutar la función asíncrona
            asyncio.run(main())
                
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        # Desconectar de IBKR
        data_manager.disconnect()