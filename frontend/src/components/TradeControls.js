import React, { useState } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Button,
    TextField,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Box,
    Slider,
    Chip,
    Alert,
    CircularProgress
} from '@mui/material';
import { PlayArrow, Stop, ShoppingCart, Sell } from '@mui/icons-material';
import axios from 'axios';
import toast from 'react-hot-toast';

const TradeControls = ({ socket }) => {
        const [symbol, setSymbol] = useState('BTCUSDT');
        const [amount, setAmount] = useState(1000);
        const [confidence, setConfidence] = useState(0.7);
        const [mode, setMode] = useState('VIRTUAL');
        const [loading, setLoading] = useState(false);
        const [autoTrade, setAutoTrade] = useState(false);

        const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'];

        const executeTrade = async(side) => {
            setLoading(true);
            try {
                const response = await axios.post('http://localhost:8000/api/v1/trade/execute', {
                    symbol,
                    investment_amount: amount,
                    auto_select_indicators: true,
                    mode: mode
                });

                if (response.data.success) {
                    toast.success(`${side} ${symbol}: ${response.data.message}`);
                    if (socket) {
                        socket.emit('trade_executed', {
                            symbol,
                            side,
                            amount,
                            price: response.data.price
                        });
                    }
                } else {
                    toast.error(response.data.message || 'Trade failed');
                }
            } catch (error) {
                console.error('Trade error:', error);
                toast.error('Failed to execute trade');
            } finally {
                setLoading(false);
            }
        };

        const toggleAutoTrade = async() => {
            setAutoTrade(!autoTrade);
            try {
                await axios.post('http://localhost:8000/api/v1/trading/auto', {
                    enabled: !autoTrade
                });
                toast.success(`Auto-trading ${!autoTrade ? 'enabled' : 'disabled'}`);
            } catch (error) {
                console.error('Auto-trade toggle error:', error);
                toast.error('Failed to toggle auto-trading');
                setAutoTrade(!autoTrade);
            }
        };

        return ( <
                Paper sx = {
                    { p: 3, mb: 3 } } >
                <
                Typography variant = "h6"
                gutterBottom > 🎮Trading Controls < /Typography> <
                Grid container spacing = { 2 }
                alignItems = "center" >
                <
                Grid item xs = { 12 }
                md = { 2 } >
                <
                FormControl fullWidth size = "small" >
                <
                InputLabel > Symbol < /InputLabel> <
                Select value = { symbol }
                onChange = {
                    (e) => setSymbol(e.target.value) } > {
                    symbols.map(s => < MenuItem key = { s }
                        value = { s } > { s } < /MenuItem>)} <
                        /Select> <
                        /FormControl> <
                        /Grid>

                        <
                        Grid item xs = { 12 }
                        md = { 2 } >
                        <
                        TextField fullWidth type = "number"
                        label = "Amount (USD)"
                        value = { amount }
                        onChange = {
                            (e) => setAmount(Number(e.target.value)) }
                        size = "small" /
                        >
                        <
                        /Grid>

                        <
                        Grid item xs = { 12 }
                        md = { 2 } >
                        <
                        FormControl fullWidth size = "small" >
                        <
                        InputLabel > Mode < /InputLabel> <
                        Select value = { mode }
                        onChange = {
                            (e) => setMode(e.target.value) } >
                        <
                        MenuItem value = "VIRTUAL" > Virtual < /MenuItem> <
                        MenuItem value = "PAPER" > Paper < /MenuItem> <
                        MenuItem value = "REAL" > Real < /MenuItem> <
                        /Select> <
                        /FormControl> <
                        /Grid>

                        <
                        Grid item xs = { 12 }
                        md = { 3 } >
                        <
                        Typography variant = "caption" > Min Confidence: {
                            (confidence * 100).toFixed(0) } % < /Typography> <
                        Slider value = { confidence }
                        onChange = {
                            (e, val) => setConfidence(val) }
                        min = { 0.5 }
                        max = { 0.95 }
                        step = { 0.05 }
                        size = "small" /
                        >
                        <
                        /Grid>

                        <
                        Grid item xs = { 12 }
                        md = { 3 } >
                        <
                        Box display = "flex"
                        gap = { 1 } >
                        <
                        Button variant = "contained"
                        color = "success"
                        onClick = {
                            () => executeTrade('BUY') }
                        disabled = { loading }
                        startIcon = { < ShoppingCart / > }
                        fullWidth >
                        {
                            loading ? < CircularProgress size = { 20 }
                            /> : 'BUY'} <
                            /Button> <
                            Button
                            variant = "contained"
                            color = "error"
                            onClick = {
                                () => executeTrade('SELL') }
                            disabled = { loading }
                            startIcon = { < Sell / > }
                            fullWidth >
                            SELL <
                            /Button> <
                            /Box> <
                            /Grid>

                            <
                            Grid item xs = { 12 } >
                            <
                            Box display = "flex"
                            justifyContent = "space-between"
                            alignItems = "center" >
                            <
                            Box >
                            <
                            Chip
                            label = { `Mode: ${mode}` }
                            color = { mode === 'REAL' ? 'error' : mode === 'PAPER' ? 'warning' : 'default' }
                            size = "small" /
                            >
                            <
                            Chip
                            label = { `Min Confidence: ${(confidence * 100).toFixed(0)}%` }
                            size = "small"
                            sx = {
                                { ml: 1 } }
                            /> <
                            /Box> <
                            Button
                            variant = { autoTrade ? "contained" : "outlined" }
                            color = { autoTrade ? "error" : "primary" }
                            onClick = { toggleAutoTrade }
                            startIcon = { autoTrade ? < Stop / > : < PlayArrow / > } >
                            { autoTrade ? 'Stop Auto-Trading' : 'Start Auto-Trading' } <
                            /Button> <
                            /Box> <
                            /Grid>

                            {
                                mode === 'REAL' && ( <
                                    Grid item xs = { 12 } >
                                    <
                                    Alert severity = "warning"
                                    sx = {
                                        { mt: 1 } } > ⚠️REAL TRADING MODE ACTIVE - Real funds will be used.Please ensure risk limits are set. <
                                    /Alert> <
                                    /Grid>
                                )
                            }

                            {
                                autoTrade && ( <
                                    Grid item xs = { 12 } >
                                    <
                                    Alert severity = "info"
                                    sx = {
                                        { mt: 1 } } > 🤖Auto - trading is active.The AI will automatically execute trades based on signals.Min confidence threshold: {
                                        (confidence * 100).toFixed(0) } %
                                    <
                                    /Alert> <
                                    /Grid>
                                )
                            } <
                            /Grid> <
                            /Paper>
                        );
                    };

                    export default TradeControls;