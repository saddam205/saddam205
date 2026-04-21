import React, { useState, useEffect } from 'react';
import {
    Grid,
    Paper,
    Card,
    CardContent,
    Typography,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Box,
    LinearProgress,
    Chip,
    Alert
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
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell
} from 'recharts';
import axios from 'axios';
import toast from 'react-hot-toast';

const Dashboard = ({ liveData, socket }) => {
        const [symbol, setSymbol] = useState('BTCUSDT');
        const [balance, setBalance] = useState(500000);
        const [positions, setPositions] = useState([]);
        const [performance, setPerformance] = useState({});
        const [priceHistory, setPriceHistory] = useState([]);
        const [signal, setSignal] = useState(null);
        const [loading, setLoading] = useState(false);

        const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'];

        useEffect(() => {
            fetchData();
            const interval = setInterval(fetchData, 5000);
            return () => clearInterval(interval);
        }, [symbol]);

        useEffect(() => {
            if (liveData) {
                setPriceHistory(prev => [...prev.slice(-50), {
                    time: new Date().toLocaleTimeString(),
                    price: liveData.price || 0
                }]);
                setSignal(liveData.signal);
            }
        }, [liveData]);

        const fetchData = async() => {
            try {
                const [balanceRes, positionsRes, perfRes, priceRes] = await Promise.all([
                    axios.get(`http://localhost:8000/api/v1/balance/VIRTUAL`),
                    axios.get(`http://localhost:8000/api/v1/positions`),
                    axios.get(`http://localhost:8000/api/v1/performance`),
                    axios.get(`http://localhost:8000/api/v1/price/${symbol}`)
                ]);

                setBalance(balanceRes.data.balance);
                setPositions(positionsRes.data.positions);
                setPerformance(perfRes.data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        const executeTrade = async(investment) => {
            setLoading(true);
            try {
                const response = await axios.post('http://localhost:8000/api/v1/trade/execute', {
                    symbol: symbol,
                    investment_amount: investment,
                    auto_select_indicators: true,
                    mode: 'VIRTUAL'
                });

                if (response.data.success) {
                    toast.success(`Trade executed: ${response.data.signal} ${symbol}`);
                    fetchData();
                } else {
                    toast.error(response.data.message);
                }
            } catch (error) {
                toast.error('Trade execution failed');
            } finally {
                setLoading(false);
            }
        };

        const COLORS = ['#00ff88', '#ff3366', '#ffaa00', '#00aaff'];

        return ( <
                Grid container spacing = { 3 } > { /* Balance Card */ } <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card sx = {
                    { background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' } } >
                <
                CardContent >
                <
                Typography variant = "h6" > Total Balance < /Typography> <
                Typography variant = "h4" > $ { balance.toLocaleString() } < /Typography> <
                Typography variant = "body2" >
                Return: { performance.total_return ? .toFixed(2) || 0 } %
                <
                /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                { /* P&L Card */ } <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card sx = {
                    { background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' } } >
                <
                CardContent >
                <
                Typography variant = "h6" > Total P & L < /Typography> <
                Typography variant = "h4"
                color = { performance.total_pnl >= 0 ? 'success.main' : 'error.main' } >
                $ { performance.total_pnl ? .toLocaleString() || 0 } <
                /Typography> <
                Typography variant = "body2" >
                Win Rate: { performance.win_rate ? .toFixed(1) || 0 } %
                <
                /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                { /* Trades Card */ } <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card >
                <
                CardContent >
                <
                Typography variant = "h6" > Total Trades < /Typography> <
                Typography variant = "h4" > { performance.total_trades || 0 } < /Typography> <
                Typography variant = "body2" >
                Wins: { performance.winning_trades || 0 } | Losses: { performance.losing_trades || 0 } <
                /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                { /* Signal Card */ } <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card sx = {
                    {
                        background: signal === 'BUY' ? 'linear-gradient(135deg, #00ff88, #00aa44)' : signal === 'SELL' ? 'linear-gradient(135deg, #ff3366, #aa1144)' : 'linear-gradient(135deg, #666, #333)'
                    }
                } >
                <
                CardContent >
                <
                Typography variant = "h6" > Current Signal < /Typography> <
                Typography variant = "h3" > { signal || 'HOLD' } < /Typography> <
                Typography variant = "body2" >
                Confidence: { liveData ? .confidence ? `${(liveData.confidence * 100).toFixed(1)}%` : '—' } <
                /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                { /* Price Chart */ } <
                Grid item xs = { 12 }
                md = { 8 } >
                <
                Paper sx = {
                    { p: 2 } } >
                <
                Typography variant = "h6" > Price Chart - { symbol } < /Typography> <
                ResponsiveContainer width = "100%"
                height = { 400 } >
                <
                AreaChart data = { priceHistory } >
                <
                defs >
                <
                linearGradient id = "colorPrice"
                x1 = "0"
                y1 = "0"
                x2 = "0"
                y2 = "1" >
                <
                stop offset = "5%"
                stopColor = "#00ff88"
                stopOpacity = { 0.8 }
                /> <
                stop offset = "95%"
                stopColor = "#00ff88"
                stopOpacity = { 0 }
                /> <
                /linearGradient> <
                /defs> <
                CartesianGrid strokeDasharray = "3 3" / >
                <
                XAxis dataKey = "time" / >
                <
                YAxis domain = {
                    ['auto', 'auto'] }
                /> <
                Tooltip / >
                <
                Area type = "monotone"
                dataKey = "price"
                stroke = "#00ff88"
                fill = "url(#colorPrice)" / >
                <
                /AreaChart> <
                /ResponsiveContainer> <
                /Paper> <
                /Grid>

                { /* Performance Metrics */ } <
                Grid item xs = { 12 }
                md = { 4 } >
                <
                Paper sx = {
                    { p: 2 } } >
                <
                Typography variant = "h6" > Performance Metrics < /Typography> <
                Box sx = {
                    { mt: 2 } } >
                <
                Typography > Sharpe Ratio: { performance.sharpe_ratio ? .toFixed(2) || '—' } < /Typography> <
                Typography > Max Drawdown: { performance.max_drawdown ? .toFixed(2) || '—' } % < /Typography> <
                Typography > Win / Loss Ratio: { performance.win_loss_ratio ? .toFixed(2) || '—' } < /Typography> <
                Typography > Profit Factor: { performance.profit_factor ? .toFixed(2) || '—' } < /Typography> <
                /Box> <
                /Paper> <
                /Grid>

                { /* Open Positions */ } <
                Grid item xs = { 12 } >
                <
                Paper sx = {
                    { p: 2 } } >
                <
                Typography variant = "h6" > Open Positions < /Typography> {
                    positions.length === 0 ? ( <
                        Typography color = "text.secondary" > No open positions < /Typography>
                    ) : (
                        positions.map(pos => ( <
                            Box key = { pos.id }
                            sx = {
                                { mb: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 } } >
                            <
                            Grid container spacing = { 2 } >
                            <
                            Grid item xs = { 3 } >
                            <
                            Typography variant = "body2" > Symbol < /Typography> <
                            Typography variant = "body1" > { pos.symbol } < /Typography> <
                            /Grid> <
                            Grid item xs = { 3 } >
                            <
                            Typography variant = "body2" > Entry Price < /Typography> <
                            Typography variant = "body1" > $ { pos.entry_price ? .toFixed(2) } < /Typography> <
                            /Grid> <
                            Grid item xs = { 3 } >
                            <
                            Typography variant = "body2" > Quantity < /Typography> <
                            Typography variant = "body1" > { pos.quantity ? .toFixed(4) } < /Typography> <
                            /Grid> <
                            Grid item xs = { 3 } >
                            <
                            Typography variant = "body2" > Current P & L < /Typography> <
                            Typography variant = "body1"
                            color = { pos.pnl >= 0 ? 'success.main' : 'error.main' } >
                            $ { pos.pnl ? .toFixed(2) } <
                            /Typography> <
                            /Grid> <
                            /Grid> <
                            /Box>
                        ))
                    )
                } <
                /Paper> <
                /Grid>

                { /* Trade Controls */ } <
                Grid item xs = { 12 } >
                <
                Paper sx = {
                    { p: 2 } } >
                <
                Typography variant = "h6" > Quick Trade < /Typography> <
                Grid container spacing = { 2 }
                sx = {
                    { mt: 1 } } >
                <
                Grid item xs = { 6 }
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
                        Grid item xs = { 6 }
                        md = { 3 } >
                        <
                        Button variant = "contained"
                        color = "success"
                        fullWidth onClick = {
                            () => executeTrade(1000) }
                        disabled = { loading } >
                        Buy $1, 000 <
                        /Button> <
                        /Grid> <
                        Grid item xs = { 6 }
                        md = { 3 } >
                        <
                        Button variant = "contained"
                        color = "error"
                        fullWidth onClick = {
                            () => executeTrade(1000) }
                        disabled = { loading } >
                        Sell $1, 000 <
                        /Button> <
                        /Grid> <
                        Grid item xs = { 6 }
                        md = { 3 } >
                        <
                        Button variant = "contained"
                        fullWidth onClick = {
                            () => executeTrade(10000) }
                        disabled = { loading } >
                        Quick Trade $10k <
                        /Button> <
                        /Grid> <
                        /Grid> {
                            loading && < LinearProgress sx = {
                                { mt: 2 } }
                            />} <
                            /Paper> <
                            /Grid> <
                            /Grid>
                        );
                    };

                    export default Dashboard;