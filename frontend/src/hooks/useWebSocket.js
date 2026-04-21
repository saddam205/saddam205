import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import useWebSocket from './useWebSocket';
const ws = new WebSocket("ws://localhost:8000/ws");
const API_BASE_URL = 'http://localhost:8000/api/v1';

const useMarketData = (symbol = 'BTCUSDT', interval = '1h') => {
    const [price, setPrice] = useState(null);
    const [historicalData, setHistoricalData] = useState([]);
    const [indicators, setIndicators] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);

    const { isConnected, lastMessage, subscribe } = useWebSocket();

    // Fetch historical data
    const fetchHistoricalData = useCallback(async() => {
        setLoading(true);
        try {
            const response = await axios.get(`${API_BASE_URL}/market/historical`, {
                params: { symbol, interval, limit: 200 }
            });

            if (response.data.success) {
                setHistoricalData(response.data.data);
                setLastUpdate(new Date());
            }
        } catch (err) {
            console.error('Failed to fetch historical data:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [symbol, interval]);

    // Fetch current price
    const fetchCurrentPrice = useCallback(async() => {
        try {
            const response = await axios.get(`${API_BASE_URL}/market/price/${symbol}`);
            if (response.data.success) {
                setPrice(response.data.price);
            }
        } catch (err) {
            console.error('Failed to fetch current price:', err);
        }
    }, [symbol]);

    // Fetch technical indicators
    const fetchIndicators = useCallback(async() => {
        try {
            const response = await axios.get(`${API_BASE_URL}/analysis/indicators/${symbol}`, {
                params: { interval, lookback: 100 }
            });
            if (response.data.success) {
                setIndicators(response.data.data);
            }
        } catch (err) {
            console.error('Failed to fetch indicators:', err);
        }
    }, [symbol, interval]);

    // Handle WebSocket messages
    useEffect(() => {
        if (lastMessage) {
            switch (lastMessage.type) {
                case 'market_update':
                    if (lastMessage.data.symbol === symbol) {
                        setPrice(lastMessage.data.price);
                        setHistoricalData(prev => {
                            const newData = [...prev, {
                                timestamp: new Date(),
                                price: lastMessage.data.price,
                                volume: lastMessage.data.volume
                            }];
                            // Keep only last 200 points
                            return newData.slice(-200);
                        });
                        setLastUpdate(new Date());
                    }
                    break;
                case 'signal_update':
                    if (lastMessage.data.symbol === symbol) {
                        setIndicators(prev => ({
                            ...prev,
                            signal: lastMessage.data.signal,
                            confidence: lastMessage.data.confidence
                        }));
                    }
                    break;
                default:
                    break;
            }
        }
    }, [lastMessage, symbol]);

    // Subscribe to market updates
    useEffect(() => {
        if (isConnected) {
            subscribe('market', symbol);
            subscribe('signals', symbol);
        }
        return () => {
            if (isConnected) {
                // Unsubscribe on cleanup
            }
        };
    }, [isConnected, symbol, subscribe]);

    // Initial data fetch
    useEffect(() => {
        fetchHistoricalData();
        fetchCurrentPrice();
        fetchIndicators();

        const intervalId = setInterval(() => {
            fetchCurrentPrice();
        }, 10000); // Update every 10 seconds

        return () => clearInterval(intervalId);
    }, [fetchHistoricalData, fetchCurrentPrice, fetchIndicators]);

    return {
        price,
        historicalData,
        indicators,
        loading,
        error,
        lastUpdate,
        isConnected,
        refresh: fetchHistoricalData,
        symbol,
        interval
    };
};

export default useMarketData;