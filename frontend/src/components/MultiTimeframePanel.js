import React, { useState } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Box,
    Chip,
    Card,
    CardContent,
    LinearProgress
} from '@mui/material';
import { TrendingUp, TrendingDown, Remove } from '@mui/icons-material';
import axios from 'axios';
import toast from 'react-hot-toast';

const MultiTimeframePanel = () => {
        const [symbol, setSymbol] = useState('BTCUSDT');
        const [loading, setLoading] = useState(false);
        const [analysis, setAnalysis] = useState(null);
        const [selectedTimeframes, setSelectedTimeframes] = useState(['1m', '5m', '15m', '1h']);

        const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'];
        const availableTimeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];

        const runAnalysis = async() => {
            setLoading(true);
            try {
                const response = await axios.post('http://localhost:8000/api/v1/analysis/multi-timeframe', {
                    symbol,
                    timeframes: selectedTimeframes,
                    lookback_bars: 200
                });

                if (response.data.success) {
                    setAnalysis(response.data.data);
                    toast.success('Multi-timeframe analysis completed');
                } else {
                    toast.error(response.data.message || 'Analysis failed');
                }
            } catch (error) {
                console.error('Analysis error:', error);
                toast.error('Failed to run analysis');
            } finally {
                setLoading(false);
            }
        };

        const getTrendIcon = (trend) => {
            if (trend === 'UP') return <TrendingUp sx = {
                { color: '#4ade80' } }
            />;
            if (trend === 'DOWN') return <TrendingDown sx = {
                { color: '#f87171' } }
            />;
            return <Remove sx = {
                { color: '#ffaa00' } }
            />;
        };

        const getSignalColor = (signal) => {
            if (signal === 'BUY') return '#4ade80';
            if (signal === 'SELL') return '#f87171';
            return '#ffaa00';
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
                gutterBottom > 🕐Multi - Timeframe Analysis < /Typography> <
                Grid container spacing = { 2 }
                sx = {
                    { mt: 1 } } >
                <
                Grid item xs = { 12 }
                md = { 4 } >
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
                        md = { 6 } >
                        <
                        FormControl fullWidth >
                        <
                        InputLabel > Timeframes < /InputLabel> <
                        Select multiple value = { selectedTimeframes }
                        onChange = {
                            (e) => setSelectedTimeframes(e.target.value) }
                        renderValue = {
                            (selected) => ( <
                                Box sx = {
                                    { display: 'flex', flexWrap: 'wrap', gap: 0.5 } } > {
                                    selected.map((value) => ( <
                                        Chip key = { value }
                                        label = { value }
                                        size = "small" / >
                                    ))
                                } <
                                /Box>
                            )
                        } >
                        {
                            availableTimeframes.map(tf => ( <
                                MenuItem key = { tf }
                                value = { tf } > { tf } <
                                /MenuItem>
                            ))
                        } <
                        /Select> <
                        /FormControl> <
                        /Grid> <
                        Grid item xs = { 12 }
                        md = { 2 } >
                        <
                        Button variant = "contained"
                        onClick = { runAnalysis }
                        disabled = { loading }
                        fullWidth sx = {
                            { height: '56px' } } >
                        { loading ? 'Analyzing...' : 'Analyze' } <
                        /Button> <
                        /Grid> <
                        /Grid> <
                        /Paper> <
                        /Grid>

                        {
                            loading && ( <
                                Grid item xs = { 12 } >
                                <
                                LinearProgress / >
                                <
                                /Grid>
                            )
                        }

                        {
                            analysis && ( <
                                >
                                <
                                Grid item xs = { 12 } >
                                <
                                Card sx = {
                                    { bgcolor: 'rgba(102, 126, 234, 0.1)' } } >
                                <
                                CardContent >
                                <
                                Typography variant = "h6"
                                align = "center"
                                gutterBottom >
                                Combined Signal <
                                /Typography> <
                                Typography variant = "h2"
                                align = "center"
                                sx = {
                                    { color: getSignalColor(analysis.combined_signal ? .signal) } } >
                                { analysis.combined_signal ? .signal || 'HOLD' } <
                                /Typography> <
                                Typography variant = "body1"
                                align = "center" >
                                Confidence: {
                                    (analysis.combined_signal ? .confidence * 100).toFixed(1) } %
                                <
                                /Typography> <
                                Grid container spacing = { 2 }
                                sx = {
                                    { mt: 2 } } >
                                <
                                Grid item xs = { 6 }
                                textAlign = "center" >
                                <
                                Typography variant = "body2" > Bullish Timeframes < /Typography> <
                                Typography variant = "h4"
                                color = "#4ade80" > { analysis.combined_signal ? .bullish_timeframes || 0 } <
                                /Typography> <
                                /Grid> <
                                Grid item xs = { 6 }
                                textAlign = "center" >
                                <
                                Typography variant = "body2" > Bearish Timeframes < /Typography> <
                                Typography variant = "h4"
                                color = "#f87171" > { analysis.combined_signal ? .bearish_timeframes || 0 } <
                                /Typography> <
                                /Grid> <
                                /Grid> <
                                /CardContent> <
                                /Card> <
                                /Grid>

                                <
                                Grid item xs = { 12 } >
                                <
                                Typography variant = "h6"
                                gutterBottom > Timeframe Analysis < /Typography> <
                                Grid container spacing = { 2 } > {
                                    Object.entries(analysis.timeframe_signals || {}).map(([tf, data]) => ( <
                                        Grid item xs = { 12 }
                                        sm = { 6 }
                                        md = { 4 }
                                        lg = { 3 }
                                        key = { tf } >
                                        <
                                        Card >
                                        <
                                        CardContent >
                                        <
                                        Box display = "flex"
                                        justifyContent = "space-between"
                                        alignItems = "center" >
                                        <
                                        Typography variant = "h6" > { tf } < /Typography> { getTrendIcon(data.trend) } <
                                        /Box> <
                                        Typography variant = "h4"
                                        align = "center"
                                        sx = {
                                            { color: getSignalColor(data.signal), my: 2 } } >
                                        { data.signal } <
                                        /Typography> <
                                        Typography variant = "body2" > Confidence: {
                                            (data.confidence * 100).toFixed(1) } % < /Typography> <
                                        Typography variant = "body2" > Trend: { data.trend } < /Typography> <
                                        Typography variant = "body2" > Momentum: { data.momentum } < /Typography> <
                                        /CardContent> <
                                        /Card> <
                                        /Grid>
                                    ))
                                } <
                                /Grid> <
                                /Grid>

                                {
                                    analysis.divergences && analysis.divergences.length > 0 && ( <
                                        Grid item xs = { 12 } >
                                        <
                                        Paper sx = {
                                            { p: 3 } } >
                                        <
                                        Typography variant = "h6"
                                        gutterBottom > ⚠️Divergences Detected < /Typography> {
                                            analysis.divergences.map((div, idx) => ( <
                                                Box key = { idx }
                                                sx = {
                                                    { mb: 1, p: 1, bgcolor: 'rgba(248, 113, 113, 0.1)', borderRadius: 1 } } >
                                                <
                                                Typography > { div.lower_tf }
                                                vs { div.higher_tf }: { div.lower_signal }
                                                vs { div.higher_signal } <
                                                /Typography> <
                                                Typography variant = "body2"
                                                color = "text.secondary" >
                                                Severity: { div.severity } <
                                                /Typography> <
                                                /Box>
                                            ))
                                        } <
                                        /Paper> <
                                        /Grid>
                                    )
                                }

                                {
                                    analysis.trend_alignment && ( <
                                        Grid item xs = { 12 } >
                                        <
                                        Paper sx = {
                                            { p: 3 } } >
                                        <
                                        Typography variant = "h6"
                                        gutterBottom > Trend Alignment < /Typography> <
                                        Typography >
                                        Aligned: { analysis.trend_alignment.aligned ? '✅ Yes' : '❌ No' } <
                                        /Typography> <
                                        Typography >
                                        Primary Trend: { analysis.trend_alignment.primary_trend } <
                                        /Typography> <
                                        Box sx = {
                                            { mt: 2 } } > {
                                            Object.entries(analysis.trend_alignment.trends_by_timeframe || {}).map(([tf, trend]) => ( <
                                                Chip key = { tf }
                                                label = { `${tf}: ${trend}` }
                                                sx = {
                                                    { m: 0.5 } }
                                                color = { trend === 'UP' ? 'success' : trend === 'DOWN' ? 'error' : 'default' }
                                                />
                                            ))
                                        } <
                                        /Box> <
                                        /Paper> <
                                        /Grid>
                                    )
                                }

                                {
                                    analysis.recommendation && ( <
                                        Grid item xs = { 12 } >
                                        <
                                        Paper sx = {
                                            { p: 3, bgcolor: 'rgba(102, 126, 234, 0.1)' } } >
                                        <
                                        Typography variant = "h6"
                                        gutterBottom > 📝Recommendation < /Typography> <
                                        Typography > { analysis.recommendation } < /Typography> <
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

                export default MultiTimeframePanel;