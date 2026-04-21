import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Card,
    CardContent,
    Box,
    Chip,
    LinearProgress,
    List,
    ListItem,
    ListItemText,
    Divider
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    Psychology,
    Newspaper,
    Twitter,
    Insights
} from '@mui/icons-material';
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar
} from 'recharts';
import axios from 'axios';

const SentimentPanel = () => {
    const [symbol, setSymbol] = useState('BTCUSDT');
    const [sentiment, setSentiment] = useState(null);
    const [loading, setLoading] = useState(true);
    const [insights, setInsights] = useState([]);
    const [sentimentHistory, setSentimentHistory] = useState([]);

    useEffect(() => {
        fetchSentiment();
        const interval = setInterval(fetchSentiment, 60000);
        return () => clearInterval(interval);
    }, [symbol]);

    const fetchSentiment = async() => {
        setLoading(true);
        try {
            const response = await axios.get(`http://localhost:8000/api/v1/sentiment/${symbol}`);
            if (response.data.success) {
                setSentiment(response.data.data);
                setInsights(response.data.data.insights || []);
                setSentimentHistory(prev => [...prev.slice(-50), {
                    timestamp: new Date(),
                    score: response.data.data.overall_score
                }]);
            }
        } catch (error) {
            console.error('Failed to fetch sentiment:', error);
        } finally {
            setLoading(false);
        }
    };

    const getSentimentColor = (score) => {
        if (score > 0.3) return '#4ade80';
        if (score < -0.3) return '#f87171';
        return '#ffaa00';
    };

    const getSentimentLabel = (score) => {
        if (score > 0.6) return 'VERY BULLISH';
        if (score > 0.3) return 'BULLISH';
        if (score > -0.3) return 'NEUTRAL';
        if (score > -0.6) return 'BEARISH';
        return 'VERY BEARISH';
    };

    const radarData = sentiment ? .source_scores ? .map(source => ({
        subject: source.source.toUpperCase(),
        score: (source.score + 1) * 50,
        fullMark: 100
    })) || [];

    if (loading && !sentiment) {
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
        gutterBottom > 🧠Market Sentiment Analysis < /Typography> <
        Typography variant = "body2"
        color = "text.secondary" >
        Multi - source sentiment aggregation with RAG - powered insights <
        /Typography> <
        /Paper> <
        /Grid>

        <
        Grid item xs = { 12 }
        md = { 4 } >
        <
        Card sx = {
            { bgcolor: 'rgba(102, 126, 234, 0.1)' } } >
        <
        CardContent >
        <
        Typography variant = "body2"
        color = "text.secondary" > Overall Sentiment < /Typography> <
        Typography variant = "h2"
        sx = {
            { color: getSentimentColor(sentiment ? .overall_score || 0) } } >
        { getSentimentLabel(sentiment ? .overall_score || 0) } <
        /Typography> <
        Typography variant = "h4" >
        Score: {
            ((sentiment ? .overall_score || 0) * 100).toFixed(1) } %
        <
        /Typography> <
        Typography variant = "body2" >
        Confidence: {
            ((sentiment ? .confidence || 0) * 100).toFixed(1) } %
        <
        /Typography> <
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
        Typography variant = "body2"
        color = "text.secondary" > Sources Analyzed < /Typography> <
        Typography variant = "h3" > { sentiment ? .sources_analyzed || 0 } < /Typography> <
        Box sx = {
            { mt: 2 } } > {
            sentiment ? .source_scores ? .map((source, idx) => ( <
                Chip key = { idx }
                icon = { source.source === 'news' ? < Newspaper / > : source.source === 'social' ? < Twitter / > : < Insights / > }
                label = { `${source.source}: ${(source.score * 100).toFixed(0)}%` }
                sx = {
                    { m: 0.5 } }
                color = { source.score > 0 ? 'success' : source.score < 0 ? 'error' : 'default' }
                />
            ))
        } <
        /Box> <
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
        Typography variant = "body2"
        color = "text.secondary" > Sentiment Shift < /Typography> {
            sentiment ? .sentiment_shift ? .shift_detected ? ( <
                >
                <
                Typography variant = "h4"
                color = { sentiment.sentiment_shift.shift_direction === 'positive' ? '#4ade80' : '#f87171' } > { sentiment.sentiment_shift.shift_direction ? .toUpperCase() }
                SHIFT <
                /Typography> <
                Typography variant = "body2" >
                Magnitude: {
                    (sentiment.sentiment_shift.shift_magnitude * 100).toFixed(1) } %
                <
                /Typography> <
                />
            ) : ( <
                Typography variant = "h4" > No Shift < /Typography>
            )
        } <
        /CardContent> <
        /Card> <
        /Grid>

        {
            radarData.length > 0 && ( <
                Grid item xs = { 12 }
                md = { 6 } >
                <
                Paper sx = {
                    { p: 3 } } >
                <
                Typography variant = "h6"
                gutterBottom > Source Sentiment Radar < /Typography> <
                ResponsiveContainer width = "100%"
                height = { 300 } >
                <
                RadarChart data = { radarData } >
                <
                PolarGrid / >
                <
                PolarAngleAxis dataKey = "subject" / >
                <
                PolarRadiusAxis domain = {
                    [0, 100] }
                /> <
                Radar name = "Sentiment"
                dataKey = "score"
                stroke = "#667eea"
                fill = "rgba(102, 126, 234, 0.5)" / >
                <
                Tooltip / >
                <
                /RadarChart> <
                /ResponsiveContainer> <
                /Paper> <
                /Grid>
            )
        }

        {
            sentimentHistory.length > 0 && ( <
                Grid item xs = { 12 }
                md = { 6 } >
                <
                Paper sx = {
                    { p: 3 } } >
                <
                Typography variant = "h6"
                gutterBottom > Sentiment History < /Typography> <
                ResponsiveContainer width = "100%"
                height = { 300 } >
                <
                AreaChart data = { sentimentHistory } >
                <
                CartesianGrid strokeDasharray = "3 3" / >
                <
                XAxis dataKey = "timestamp"
                tickFormatter = {
                    (t) => new Date(t).toLocaleTimeString() }
                /> <
                YAxis domain = {
                    [-1, 1] }
                /> <
                Tooltip / >
                <
                Area type = "monotone"
                dataKey = "score"
                stroke = "#667eea"
                fill = "rgba(102, 126, 234, 0.1)" /
                >
                <
                /AreaChart> <
                /ResponsiveContainer> <
                /Paper> <
                /Grid>
            )
        }

        {
            insights.length > 0 && ( <
                Grid item xs = { 12 } >
                <
                Paper sx = {
                    { p: 3 } } >
                <
                Typography variant = "h6"
                gutterBottom >
                <
                Psychology sx = {
                    { mr: 1, verticalAlign: 'middle' } }
                />
                AI - Powered Market Insights <
                /Typography> <
                List > {
                    insights.map((insight, idx) => ( <
                        React.Fragment key = { idx } >
                        <
                        ListItem >
                        <
                        ListItemText primary = { insight }
                        secondary = { `Confidence: ${(sentiment?.confidence * 100).toFixed(0)}%` }
                        /> <
                        /ListItem> { idx < insights.length - 1 && < Divider / > } <
                        /React.Fragment>
                    ))
                } <
                /List> <
                /Paper> <
                /Grid>
            )
        }

        <
        Grid item xs = { 12 } >
        <
        Paper sx = {
            { p: 3, bgcolor: 'rgba(102, 126, 234, 0.05)' } } >
        <
        Typography variant = "body2"
        color = "text.secondary" > 💡Sentiment analysis combines news articles, social media, on - chain data, and technical indicators.RAG system retrieves relevant market intelligence to provide contextual insights. <
        /Typography> <
        /Paper> <
        /Grid> <
        /Grid>
    );
};

export default SentimentPanel;