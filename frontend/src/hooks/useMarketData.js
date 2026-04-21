import { useState, useEffect, useCallback, useRef } from 'react';
import io from 'socket.io-client';

const useWebSocket = (url = 'http://localhost:8000/ws', options = {}) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const [error, setError] = useState(null);
    const socketRef = useRef(null);
    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = options.maxReconnectAttempts || 5;

    const connect = useCallback(() => {
        if (socketRef.current ? .connected) return;

        socketRef.current = io(url, {
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: maxReconnectAttempts,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            ...options
        });

        socketRef.current.on('connect', () => {
            console.log('WebSocket connected');
            setIsConnected(true);
            setError(null);
            reconnectAttempts.current = 0;
        });

        socketRef.current.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            setIsConnected(false);
            if (reason === 'io server disconnect') {
                // Reconnect manually
                socketRef.current.connect();
            }
        });

        socketRef.current.on('connect_error', (err) => {
            console.error('WebSocket connection error:', err);
            setError(err.message);
            setIsConnected(false);
            reconnectAttempts.current++;

            if (reconnectAttempts.current >= maxReconnectAttempts) {
                console.error('Max reconnection attempts reached');
                socketRef.current ? .close();
            }
        });

        // Handle specific message types
        socketRef.current.on('market_update', (data) => {
            setLastMessage({ type: 'market_update', data });
        });

        socketRef.current.on('signal_update', (data) => {
            setLastMessage({ type: 'signal_update', data });
        });

        socketRef.current.on('trade_executed', (data) => {
            setLastMessage({ type: 'trade_executed', data });
        });

        socketRef.current.on('portfolio_update', (data) => {
            setLastMessage({ type: 'portfolio_update', data });
        });

        socketRef.current.on('error', (data) => {
            setError(data.message);
            setLastMessage({ type: 'error', data });
        });
    }, [url, options, maxReconnectAttempts]);

    const disconnect = useCallback(() => {
        if (socketRef.current) {
            socketRef.current.disconnect();
            socketRef.current = null;
            setIsConnected(false);
        }
    }, []);

    const sendMessage = useCallback((event, data) => {
        if (socketRef.current ? .connected) {
            socketRef.current.emit(event, data);
            return true;
        }
        console.warn('WebSocket not connected, message not sent');
        return false;
    }, []);

    const subscribe = useCallback((channel, symbol = null) => {
        sendMessage('subscribe', { channel, symbol });
    }, [sendMessage]);

    const unsubscribe = useCallback((channel, symbol = null) => {
        sendMessage('unsubscribe', { channel, symbol });
    }, [sendMessage]);

    useEffect(() => {
        connect();
        return () => {
            disconnect();
        };
    }, [connect, disconnect]);

    return {
        isConnected,
        lastMessage,
        error,
        sendMessage,
        subscribe,
        unsubscribe,
        connect,
        disconnect
    };
};

export default useWebSocket;