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
    CircularProgress,
    Alert,
    Card,
    CardContent
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area } from 'recharts';
import axios from 'axios';
import toast from 'react-hot-toast';

const BacktestPanel = () => {
        const [symbol, setSymbol] = useState('BTCUSDT');
        const [strategy, setStrategy] = useState('trend_following');
        const [startDate, setStartDate] = useState('2024-01-01');
        const [endDate, setEndDate] = useState('2024-12-31');
        const [initialCapital, setInitialCapital] = useState(100000);
        const [loading, setLoading] = useState(false);
        const [results, setResults] = useState(null);
        const [equityCurve, setEquityCurve] = useState([]);

        const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'];
        const strategies = ['trend_following', 'mean_reversion', 'momentum', 'breakout'];

        const runBacktest = async() => {
            setLoading(true);
            try {
                const response = await axios.post('http://localhost:8000/api/v1/backtest/run', {
                    symbol,
                    strategy,
                    start_date: startDate,
                    end_date: endDate,
                    initial_capital: initialCapital
                });

                if (response.data.success) {
                    setResults(response.data.results);
                    setEquityCurve(response.data.equity_curve || []);
                    toast.success('Backtest completed successfully');
                } else {
                    toast.error(response.data.message || 'Backtest failed');
                }
            } catch (error) {
                console.error('Backtest error:', error);
                toast.error('Failed to run backtest');
            } finally {
                setLoading(false);
            }
        };

        return ( <
                Grid container spacing = { 3 } >
                <
                Grid item xs = { 12 } >
                <
                Paper sx = {
                    { p: 3 } } >
                <
                Typography variant = "h5"
                gutterBottom > 📊Backtest Configuration < /Typography> <
                Grid container spacing = { 2 }
                sx = {
                    { mt: 1 } } >
                <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                FormControl fullWidth >
                <
                InputLabel > Symbol < /InputLabel> <
                Select value = { symbol }
                onChange = {
                    (e) => setSymbol(e.target.value) } > {
                    symbols.map(s => < MenuItem key = { s }
                        value = { s } > { s } < /MenuItem>)} <
                        /Select> <
                        /FormControl> <
                        /Grid> <
                        Grid item xs = { 12 }
                        md = { 3 } >
                        <
                        FormControl fullWidth >
                        <
                        InputLabel > Strategy < /InputLabel> <
                        Select value = { strategy }
                        onChange = {
                            (e) => setStrategy(e.target.value) } > {
                            strategies.map(s => < MenuItem key = { s }
                                value = { s } > { s.replace('_', ' ').toUpperCase() } < /MenuItem>)} <
                                /Select> <
                                /FormControl> <
                                /Grid> <
                                Grid item xs = { 12 }
                                md = { 2 } >
                                <
                                TextField fullWidth type = "date"
                                label = "Start Date"
                                value = { startDate }
                                onChange = {
                                    (e) => setStartDate(e.target.value) }
                                InputLabelProps = {
                                    { shrink: true } }
                                /> <
                                /Grid> <
                                Grid item xs = { 12 }
                                md = { 2 } >
                                <
                                TextField fullWidth type = "date"
                                label = "End Date"
                                value = { endDate }
                                onChange = {
                                    (e) => setEndDate(e.target.value) }
                                InputLabelProps = {
                                    { shrink: true } }
                                /> <
                                /Grid> <
                                Grid item xs = { 12 }
                                md = { 2 } >
                                <
                                TextField fullWidth type = "number"
                                label = "Initial Capital"
                                value = { initialCapital }
                                onChange = {
                                    (e) => setInitialCapital(Number(e.target.value)) }
                                /> <
                                /Grid> <
                                /Grid> <
                                Button variant = "contained"
                                onClick = { runBacktest }
                                disabled = { loading }
                                sx = {
                                    { mt: 3 } }
                                fullWidth >
                                {
                                    loading ? < CircularProgress size = { 24 }
                                    /> : 'Run Backtest'} <
                                    /Button> <
                                    /Paper> <
                                    /Grid>

                                    {
                                        results && ( <
                                            >
                                            <
                                            Grid item xs = { 12 }
                                            md = { 4 } >
                                            <
                                            Card >
                                            <
                                            CardContent >
                                            <
                                            Typography variant = "h6" > Returns < /Typography> <
                                            Typography variant = "h4"
                                            color = { results.total_return >= 0 ? 'success.main' : 'error.main' } > { results.total_return ? .toFixed(2) } %
                                            <
                                            /Typography> <
                                            Typography variant = "body2" > Initial: $ { results.initial_capital ? .toLocaleString() } < /Typography> <
                                            Typography variant = "body2" > Final: $ { results.final_capital ? .toLocaleString() } < /Typography> <
                                            /CardContent> <
                                            /Card> <
                                            /Grid>

                                            <
                                            Grid item xs = { 12 }
                                            md = { 4 } >
                                            <
                                            Card >
                                            <
                                            CardContent >
                                            <
                                            Typography variant = "h6" > Risk Metrics < /Typography> <
                                            Typography variant = "body1" > Sharpe Ratio: { results.sharpe_ratio ? .toFixed(2) } < /Typography> <
                                            Typography variant = "body1" > Max Drawdown: { results.max_drawdown ? .toFixed(2) } % < /Typography> <
                                            Typography variant = "body1" > Win Rate: { results.win_rate ? .toFixed(1) } % < /Typography> <
                                            /CardContent> <
                                            /Card> <
                                            /Grid>

                                            <
                                            Grid item xs = { 12 }
                                            md = { 4 } >
                                            <
                                            Card >
                                            <
                                            CardContent >
                                            <
                                            Typography variant = "h6" > Trade Statistics < /Typography> <
                                            Typography variant = "body1" > Total Trades: { results.total_trades } < /Typography> <
                                            Typography variant = "body1" > Winning Trades: { results.winning_trades } < /Typography> <
                                            Typography variant = "body1" > Losing Trades: { results.losing_trades } < /Typography> <
                                            Typography variant = "body1" > Profit Factor: { results.profit_factor ? .toFixed(2) } < /Typography> <
                                            /CardContent> <
                                            /Card> <
                                            /Grid>

                                            {
                                                equityCurve.length > 0 && ( <
                                                    Grid item xs = { 12 } >
                                                    <
                                                    Paper sx = {
                                                        { p: 3 } } >
                                                    <
                                                    Typography variant = "h6"
                                                    gutterBottom > Equity Curve < /Typography> <
                                                    ResponsiveContainer width = "100%"
                                                    height = { 400 } >
                                                    <
                                                    LineChart data = { equityCurve } >
                                                    <
                                                    CartesianGrid strokeDasharray = "3 3" / >
                                                    <
                                                    XAxis dataKey = "timestamp" / >
                                                    <
                                                    YAxis domain = {
                                                        ['auto', 'auto'] }
                                                    /> <
                                                    Tooltip / >
                                                    <
                                                    Area type = "monotone"
                                                    dataKey = "equity"
                                                    stroke = "#667eea"
                                                    fill = "rgba(102, 126, 234, 0.1)" / >
                                                    <
                                                    /LineChart> <
                                                    /ResponsiveContainer> <
                                                    /Paper> <
                                                    /Grid>
                                                )
                                            } <
                                            />
                                        )
                                    } <
                                    /Grid>
                                );
                            };

                            export default BacktestPanel;