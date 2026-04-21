import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Box
} from '@mui/material';
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ComposedChart
} from 'recharts';
import axios from 'axios';

const Charts = ({ liveData }) => {
        const [timeframe, setTimeframe] = useState('1h');
        const [historicalData, setHistoricalData] = useState([]);
        const [indicators, setIndicators] = useState({
            sma20: [],
            sma50: [],
            rsi: []
        });
        const [loading, setLoading] = useState(false);

        const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

        useEffect(() => {
            fetchHistoricalData();
        }, [timeframe]);

        const fetchHistoricalData = async() => {
            setLoading(true);
            try {
                const response = await axios.get(`http://localhost:8000/api/v1/analysis/historical/BTCUSDT`, {
                    params: { interval: timeframe, limit: 100 }
                });

                if (response.data.success) {
                    const data = response.data.data.map(item => ({
                        time: new Date(item.timestamp).toLocaleTimeString(),
                        price: item.close,
                        volume: item.volume,
                        high: item.high,
                        low: item.low
                    }));
                    setHistoricalData(data);
                    calculateIndicators(data);
                }
            } catch (error) {
                console.error('Failed to fetch historical data:', error);
            } finally {
                setLoading(false);
            }
        };

        const calculateIndicators = (data) => {
            // Simple SMA calculation
            const prices = data.map(d => d.price);
            const sma20 = [];
            const sma50 = [];

            for (let i = 0; i < prices.length; i++) {
                if (i >= 19) {
                    const sum20 = prices.slice(i - 19, i + 1).reduce((a, b) => a + b, 0);
                    sma20.push({ time: data[i].time, value: sum20 / 20 });
                } else {
                    sma20.push({ time: data[i].time, value: null });
                }

                if (i >= 49) {
                    const sum50 = prices.slice(i - 49, i + 1).reduce((a, b) => a + b, 0);
                    sma50.push({ time: data[i].time, value: sum50 / 50 });
                } else {
                    sma50.push({ time: data[i].time, value: null });
                }
            }

            setIndicators({ sma20, sma50 });
        };

        return ( <
                Grid container spacing = { 3 } >
                <
                Grid item xs = { 12 } >
                <
                Paper sx = {
                    { p: 3 } } >
                <
                Box display = "flex"
                justifyContent = "space-between"
                alignItems = "center"
                mb = { 2 } >
                <
                Typography variant = "h5" > 📈Price Chart < /Typography> <
                FormControl sx = {
                    { minWidth: 120 } } >
                <
                InputLabel > Timeframe < /InputLabel> <
                Select value = { timeframe }
                onChange = {
                    (e) => setTimeframe(e.target.value) } > {
                    timeframes.map(tf => < MenuItem key = { tf }
                        value = { tf } > { tf } < /MenuItem>)} <
                        /Select> <
                        /FormControl> <
                        /Box>

                        <
                        ResponsiveContainer width = "100%"
                        height = { 500 } >
                        <
                        ComposedChart data = { historicalData } >
                        <
                        CartesianGrid strokeDasharray = "3 3" / >
                        <
                        XAxis dataKey = "time" / >
                        <
                        YAxis yAxisId = "left"
                        domain = {
                            ['auto', 'auto'] }
                        /> <
                        YAxis yAxisId = "right"
                        orientation = "right"
                        domain = {
                            ['auto', 'auto'] }
                        /> <
                        Tooltip / >
                        <
                        Legend / >
                        <
                        Area yAxisId = "left"
                        type = "monotone"
                        dataKey = "price"
                        stroke = "#667eea"
                        fill = "rgba(102, 126, 234, 0.1)"
                        name = "Price" /
                        >
                        <
                        Line yAxisId = "left"
                        type = "monotone"
                        dataKey = { indicators.sma20.map(d => d.value) }
                        stroke = "#4ade80"
                        dot = { false }
                        name = "SMA 20" /
                        >
                        <
                        Line yAxisId = "left"
                        type = "monotone"
                        dataKey = { indicators.sma50.map(d => d.value) }
                        stroke = "#f87171"
                        dot = { false }
                        name = "SMA 50" /
                        >
                        <
                        Bar yAxisId = "right"
                        dataKey = "volume"
                        fill = "#764ba2"
                        opacity = { 0.5 }
                        name = "Volume" /
                        >
                        <
                        /ComposedChart> <
                        /ResponsiveContainer> <
                        /Paper> <
                        /Grid>

                        <
                        Grid item xs = { 12 }
                        md = { 6 } >
                        <
                        Paper sx = {
                            { p: 3 } } >
                        <
                        Typography variant = "h6"
                        gutterBottom > Volume Profile < /Typography> <
                        ResponsiveContainer width = "100%"
                        height = { 300 } >
                        <
                        BarChart data = { historicalData.slice(-30) } >
                        <
                        CartesianGrid strokeDasharray = "3 3" / >
                        <
                        XAxis dataKey = "time" / >
                        <
                        YAxis / >
                        <
                        Tooltip / >
                        <
                        Bar dataKey = "volume"
                        fill = "#667eea" / >
                        <
                        /BarChart> <
                        /ResponsiveContainer> <
                        /Paper> <
                        /Grid>

                        <
                        Grid item xs = { 12 }
                        md = { 6 } >
                        <
                        Paper sx = {
                            { p: 3 } } >
                        <
                        Typography variant = "h6"
                        gutterBottom > Price Distribution < /Typography> <
                        ResponsiveContainer width = "100%"
                        height = { 300 } >
                        <
                        LineChart data = { historicalData.slice(-50) } >
                        <
                        CartesianGrid strokeDasharray = "3 3" / >
                        <
                        XAxis dataKey = "time" / >
                        <
                        YAxis domain = {
                            ['auto', 'auto'] }
                        /> <
                        Tooltip / >
                        <
                        Line type = "monotone"
                        dataKey = "high"
                        stroke = "#4ade80"
                        name = "High" / >
                        <
                        Line type = "monotone"
                        dataKey = "low"
                        stroke = "#f87171"
                        name = "Low" / >
                        <
                        Line type = "monotone"
                        dataKey = "price"
                        stroke = "#667eea"
                        name = "Close"
                        strokeWidth = { 2 }
                        /> <
                        /LineChart> <
                        /ResponsiveContainer> <
                        /Paper> <
                        /Grid>

                        {
                            liveData && liveData.price > 0 && ( <
                                Grid item xs = { 12 } >
                                <
                                Paper sx = {
                                    { p: 3, bgcolor: 'rgba(102, 126, 234, 0.1)' } } >
                                <
                                Typography variant = "h6"
                                gutterBottom > Live Market Data < /Typography> <
                                Grid container spacing = { 2 } >
                                <
                                Grid item xs = { 3 } >
                                <
                                Typography variant = "body2"
                                color = "text.secondary" > Current Price < /Typography> <
                                Typography variant = "h5" > $ { liveData.price.toLocaleString() } < /Typography> <
                                /Grid> <
                                Grid item xs = { 3 } >
                                <
                                Typography variant = "body2"
                                color = "text.secondary" > Signal < /Typography> <
                                Typography variant = "h5"
                                color = { liveData.signal === 'BUY' ? '#4ade80' : liveData.signal === 'SELL' ? '#f87171' : '#ffaa00' } > { liveData.signal } <
                                /Typography> <
                                /Grid> <
                                Grid item xs = { 3 } >
                                <
                                Typography variant = "body2"
                                color = "text.secondary" > Confidence < /Typography> <
                                Typography variant = "h5" > {
                                    (liveData.confidence * 100).toFixed(1) } % < /Typography> <
                                /Grid> <
                                Grid item xs = { 3 } >
                                <
                                Typography variant = "body2"
                                color = "text.secondary" > Last Update < /Typography> <
                                Typography variant = "body1" > { liveData.timestamp ? .toLocaleTimeString() || '--' } < /Typography> <
                                /Grid> <
                                /Grid> <
                                /Paper> <
                                /Grid>
                            )
                        } <
                        /Grid>
                    );
                };

                export default Charts;