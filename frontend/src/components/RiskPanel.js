import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Card,
    CardContent,
    Box,
    LinearProgress,
    Chip,
    Alert
} from '@mui/material';
import { Warning, CheckCircle, Error } from '@mui/icons-material';
import {
    PieChart,
    Pie,
    Cell,
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend
} from 'recharts';
import axios from 'axios';

const RiskPanel = () => {
        const [riskMetrics, setRiskMetrics] = useState(null);
        const [positions, setPositions] = useState([]);
        const [loading, setLoading] = useState(true);
        const [limits, setLimits] = useState({
            max_position_pct: 10,
            max_daily_loss: 5,
            max_drawdown: 15,
            max_consecutive_losses: 5
        });

        useEffect(() => {
            fetchRiskData();
            const interval = setInterval(fetchRiskData, 30000);
            return () => clearInterval(interval);
        }, []);

        const fetchRiskData = async() => {
            try {
                const [metricsRes, positionsRes] = await Promise.all([
                    axios.get('http://localhost:8000/api/v1/risk/metrics'),
                    axios.get('http://localhost:8000/api/v1/positions')
                ]);

                setRiskMetrics(metricsRes.data);
                setPositions(positionsRes.data.positions || []);
            } catch (error) {
                console.error('Failed to fetch risk data:', error);
            } finally {
                setLoading(false);
            }
        };

        const getRiskColor = (value, threshold) => {
            if (value >= threshold) return '#f87171';
            if (value >= threshold * 0.7) return '#ffaa00';
            return '#4ade80';
        };

        const exposureData = positions.map(pos => ({
            name: pos.symbol,
            value: pos.value || 0,
            pnl: pos.pnl || 0
        }));

        const COLORS = ['#667eea', '#764ba2', '#4ade80', '#f87171', '#ffaa00'];

        if (loading) {
            return <LinearProgress / > ;
        }

        return ( <
                Grid container spacing = { 3 } >
                <
                Grid item xs = { 12 } >
                <
                Paper sx = {
                    { p: 3 } } >
                <
                Typography variant = "h5"
                gutterBottom > 🛡️Risk Management Dashboard < /Typography> <
                Typography variant = "body2"
                color = "text.secondary" >
                Real - time risk monitoring and position management <
                /Typography> <
                /Paper> <
                /Grid>

                <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card >
                <
                CardContent >
                <
                Typography variant = "body2"
                color = "text.secondary" > Portfolio Exposure < /Typography> <
                Typography variant = "h4" > { riskMetrics ? .exposure_ratio ? .toFixed(1) || 0 } %
                <
                /Typography> <
                LinearProgress variant = "determinate"
                value = { riskMetrics ? .exposure_ratio || 0 }
                sx = {
                    {
                        mt: 1,
                        height: 8,
                        borderRadius: 4,
                        bgcolor: '#333',
                        '& .MuiLinearProgress-bar': {
                            bgcolor: getRiskColor(riskMetrics ? .exposure_ratio || 0, limits.max_position_pct)
                        }
                    }
                }
                /> <
                Typography variant = "caption"
                color = "text.secondary" >
                Limit: { limits.max_position_pct } %
                <
                /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card >
                <
                CardContent >
                <
                Typography variant = "body2"
                color = "text.secondary" > Current Drawdown < /Typography> <
                Typography variant = "h4"
                color = { riskMetrics ? .current_drawdown > 10 ? 'error.main' : 'warning.main' } > { riskMetrics ? .current_drawdown ? .toFixed(1) || 0 } %
                <
                /Typography> <
                LinearProgress variant = "determinate"
                value = { Math.min((riskMetrics ? .current_drawdown || 0) / limits.max_drawdown * 100, 100) }
                sx = {
                    {
                        mt: 1,
                        height: 8,
                        borderRadius: 4,
                        bgcolor: '#333',
                        '& .MuiLinearProgress-bar': {
                            bgcolor: getRiskColor(riskMetrics ? .current_drawdown || 0, limits.max_drawdown)
                        }
                    }
                }
                /> <
                Typography variant = "caption"
                color = "text.secondary" >
                Max Allowed: { limits.max_drawdown } %
                <
                /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card >
                <
                CardContent >
                <
                Typography variant = "body2"
                color = "text.secondary" > Value at Risk(95 % ) < /Typography> <
                Typography variant = "h4"
                color = "error.main" > { riskMetrics ? .var_95 ? .toFixed(1) || 0 } %
                <
                /Typography> <
                Typography variant = "body2" > Conditional VaR: { riskMetrics ? .cvar_95 ? .toFixed(1) || 0 } % < /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                <
                Grid item xs = { 12 }
                md = { 3 } >
                <
                Card >
                <
                CardContent >
                <
                Typography variant = "body2"
                color = "text.secondary" > Consecutive Losses < /Typography> <
                Typography variant = "h4"
                color = { riskMetrics ? .consecutive_losses >= limits.max_consecutive_losses ? 'error.main' : 'warning.main' } > { riskMetrics ? .consecutive_losses || 0 } <
                /Typography> <
                Typography variant = "caption"
                color = "text.secondary" >
                Limit: { limits.max_consecutive_losses } <
                /Typography> <
                /CardContent> <
                /Card> <
                /Grid>

                {
                    exposureData.length > 0 && ( <
                        Grid item xs = { 12 }
                        md = { 6 } >
                        <
                        Paper sx = {
                            { p: 3 } } >
                        <
                        Typography variant = "h6"
                        gutterBottom > Portfolio Allocation < /Typography> <
                        ResponsiveContainer width = "100%"
                        height = { 300 } >
                        <
                        PieChart >
                        <
                        Pie data = { exposureData }
                        cx = "50%"
                        cy = "50%"
                        labelLine = { false }
                        label = {
                            ({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%` }
                        outerRadius = { 80 }
                        fill = "#8884d8"
                        dataKey = "value" >
                        {
                            exposureData.map((entry, index) => ( <
                                Cell key = { `cell-${index}` }
                                fill = { COLORS[index % COLORS.length] }
                                />
                            ))
                        } <
                        /Pie> <
                        Tooltip / >
                        <
                        /PieChart> <
                        /ResponsiveContainer> <
                        /Paper> <
                        /Grid>
                    )
                }

                <
                Grid item xs = { 12 }
                md = { 6 } >
                <
                Paper sx = {
                    { p: 3 } } >
                <
                Typography variant = "h6"
                gutterBottom > Risk Limits Status < /Typography> {
                    Object.entries(limits).map(([limit, value]) => {
                            const currentValue = riskMetrics ? .[limit] || 0;
                            const status = currentValue >= value ? 'exceeded' : currentValue >= value * 0.7 ? 'warning' : 'ok';
                            return ( <
                                    Box key = { limit }
                                    sx = {
                                        { mb: 2 } } >
                                    <
                                    Box display = "flex"
                                    justifyContent = "space-between" >
                                    <
                                    Typography variant = "body2" > { limit.replace(/_/g, ' ').toUpperCase() } <
                                    /Typography> <
                                    Typography variant = "body2" > { currentValue.toFixed(1) }
                                    / {value} <
                                    /Typography> {
                                        status === 'exceeded' && < Error sx = {
                                            { color: '#f87171', fontSize: 16 } }
                                        />} {
                                            status === 'warning' && < Warning sx = {
                                                { color: '#ffaa00', fontSize: 16 } }
                                            />} {
                                                status === 'ok' && < CheckCircle sx = {
                                                    { color: '#4ade80', fontSize: 16 } }
                                                />} <
                                                /Box> <
                                                LinearProgress
                                                variant = "determinate"
                                                value = { Math.min((currentValue / value) * 100, 100) }
                                                sx = {
                                                    {
                                                        height: 6,
                                                        borderRadius: 3,
                                                        bgcolor: '#333',
                                                        '& .MuiLinearProgress-bar': {
                                                            bgcolor: status === 'exceeded' ? '#f87171' : status === 'warning' ? '#ffaa00' : '#4ade80'
                                                        }
                                                    }
                                                }
                                                /> <
                                                /Box>
                                            );
                                        })
                                } <
                                /Paper> <
                                /Grid>

                            {
                                riskMetrics ? .daily_pnl && riskMetrics.daily_pnl.length > 0 && ( <
                                    Grid item xs = { 12 } >
                                    <
                                    Paper sx = {
                                        { p: 3 } } >
                                    <
                                    Typography variant = "h6"
                                    gutterBottom > Daily P & L History < /Typography> <
                                    ResponsiveContainer width = "100%"
                                    height = { 300 } >
                                    <
                                    BarChart data = { riskMetrics.daily_pnl.slice(-30) } >
                                    <
                                    CartesianGrid strokeDasharray = "3 3" / >
                                    <
                                    XAxis dataKey = "date" / >
                                    <
                                    YAxis / >
                                    <
                                    Tooltip / >
                                    <
                                    Bar dataKey = "pnl"
                                    fill = "#667eea" > {
                                        riskMetrics.daily_pnl.slice(-30).map((entry, index) => ( <
                                            Cell key = { `cell-${index}` }
                                            fill = { entry.pnl >= 0 ? '#4ade80' : '#f87171' }
                                            />
                                        ))
                                    } <
                                    /Bar> <
                                    /BarChart> <
                                    /ResponsiveContainer> <
                                    /Paper> <
                                    /Grid>
                                )
                            }

                            {
                                riskMetrics ? .limits && ( <
                                    Grid item xs = { 12 } >
                                    <
                                    Alert severity = "info"
                                    sx = {
                                        { bgcolor: 'rgba(102, 126, 234, 0.1)' } } >
                                    <
                                    Typography variant = "body2" >
                                    Risk limits are automatically enforced.Trading will be paused
                                    if any limit is exceeded.Current risk score: { riskMetrics.risk_score ? .toFixed(0) || 0 }
                                    /100 <
                                    /Typography> <
                                    /Alert> <
                                    /Grid>
                                )
                            } <
                            /Grid>
                        );
                    };

                    export default RiskPanel;