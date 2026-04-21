import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]
        self.signal_weights = {
            Timeframe.M1: 0.1,
            Timeframe.M5: 0.2,
            Timeframe.M15: 0.3,
            Timeframe.H1: 0.4
        }
        
    def analyze_all_timeframes(self, data_dict: Dict[Timeframe, pd.DataFrame]) -> Dict:
        """
        Analyze signals across multiple timeframes
        
        Args:
            data_dict: Dictionary with timeframe as key and OHLCV data as value
        """
        results = {}
        signals = {}
        
        # Analyze each timeframe
        for tf, data in data_dict.items():
            tf_signals = self._analyze_single_timeframe(data, tf)
            signals[tf] = tf_signals
            results[tf.value] = tf_signals
        
        # Combine signals
        combined_signal = self._combine_signals(signals)
        results['combined'] = combined_signal
        
        # Detect divergences
        divergences = self._detect_divergences(signals)
        results['divergences'] = divergences
        
        # Trend alignment
        alignment = self._check_trend_alignment(data_dict)
        results['trend_alignment'] = alignment
        
        return results
    
    def _analyze_single_timeframe(self, data: pd.DataFrame, tf: Timeframe) -> Dict:
        """Analyze single timeframe"""
        last_row = data.iloc[-1]
        
        # Technical indicators
        sma_20 = last_row.get('SMA_20', last_row['close'])
        sma_50 = last_row.get('SMA_50', last_row['close'])
        rsi = last_row.get('RSI', 50)
        macd = last_row.get('MACD', 0)
        macd_signal = last_row.get('MACD_Signal', 0)
        
        # Determine signal
        trend = "UP" if sma_20 > sma_50 else "DOWN"
        momentum = "BULLISH" if macd > macd_signal else "BEARISH"
        strength = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        
        # Signal score (-1 to 1)
        signal_score = 0
        if trend == "UP":
            signal_score += 0.3
        else:
            signal_score -= 0.3
            
        if momentum == "BULLISH":
            signal_score += 0.3
        else:
            signal_score -= 0.3
            
        if strength == "OVERSOLD":
            signal_score += 0.2
        elif strength == "OVERBOUGHT":
            signal_score -= 0.2
        
        # Final signal
        if signal_score > 0.3:
            final_signal = "BUY"
            confidence = min(0.5 + signal_score, 0.9)
        elif signal_score < -0.3:
            final_signal = "SELL"
            confidence = min(0.5 + abs(signal_score), 0.9)
        else:
            final_signal = "HOLD"
            confidence = 0.5
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'signal_score': signal_score,
            'trend': trend,
            'momentum': momentum,
            'strength': strength,
            'rsi': rsi,
            'price': last_row['close']
        }
    
    def _combine_signals(self, signals: Dict[Timeframe, Dict]) -> Dict:
        """Combine signals from all timeframes"""
        total_score = 0
        total_confidence = 0
        weight_sum = 0
        
        for tf, signal_data in signals.items():
            weight = self.signal_weights.get(tf, 0.25)
            weight_sum += weight
            
            # Convert signal to numeric score
            if signal_data['signal'] == "BUY":
                score = 1
            elif signal_data['signal'] == "SELL":
                score = -1
            else:
                score = 0
                
            total_score += score * weight * signal_data['confidence']
            total_confidence += signal_data['confidence'] * weight
        
        avg_confidence = total_confidence / weight_sum if weight_sum > 0 else 0
        
        # Determine final signal
        if total_score > 0.3:
            final_signal = "BUY"
        elif total_score < -0.3:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
        
        return {
            'signal': final_signal,
            'confidence': avg_confidence,
            'aggregate_score': total_score,
            'timeframes_analyzed': len(signals),
            'bullish_timeframes': sum(1 for s in signals.values() if s['signal'] == "BUY"),
            'bearish_timeframes': sum(1 for s in signals.values() if s['signal'] == "SELL")
        }
    
    def _detect_divergences(self, signals: Dict[Timeframe, Dict]) -> List[Dict]:
        """Detect divergences between timeframes"""
        divergences = []
        
        # Compare each timeframe with the next higher timeframe
        tf_list = list(signals.keys())
        for i in range(len(tf_list) - 1):
            lower_tf = tf_list[i]
            higher_tf = tf_list[i + 1]
            
            lower_signal = signals[lower_tf]['signal']
            higher_signal = signals[higher_tf]['signal']
            
            if lower_signal != higher_signal and lower_signal != "HOLD" and higher_signal != "HOLD":
                divergences.append({
                    'type': 'TIMEFRAME_DIVERGENCE',
                    'lower_tf': lower_tf.value,
                    'higher_tf': higher_tf.value,
                    'lower_signal': lower_signal,
                    'higher_signal': higher_signal,
                    'severity': 'HIGH' if lower_signal != higher_signal else 'MEDIUM'
                })
        
        return divergences
    
    def _check_trend_alignment(self, data_dict: Dict[Timeframe, pd.DataFrame]) -> Dict:
        """Check trend alignment across timeframes"""
        trends = {}
        
        for tf, data in data_dict.items():
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            trends[tf.value] = "UP" if sma_20 > sma_50 else "DOWN"
        
        # Check if all trends align
        unique_trends = set(trends.values())
        is_aligned = len(unique_trends) == 1
        
        return {
            'aligned': is_aligned,
            'primary_trend': unique_trends.pop() if is_aligned else 'MIXED',
            'trends_by_timeframe': trends
        }