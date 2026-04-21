import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';

// Mock the API calls
jest.mock('../services/api', () => ({
  getStatus: jest.fn(),
  getPortfolio: jest.fn(),
  getStrategies: jest.fn(),
  executeTrade: jest.fn(),
}));

import api from '../services/api';

describe('App Component', () => {
  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    render(<App />);
    expect(screen.getByText(/Loading/i)).toBeInTheDocument();
  });

  test('loads and displays dashboard data', async () => {
    // Mock API responses
    api.getStatus.mockResolvedValue({
      status: 'running',
      portfolio_value: 100000,
      active_strategies: ['momentum', 'mean_reversion']
    });
    
    api.getPortfolio.mockResolvedValue({
      total_value: 100000,
      cash: 75000,
      positions: [{ symbol: 'BTC/USDT', quantity: 0.5, value: 25000 }]
    });
    
    api.getStrategies.mockResolvedValue({
      strategies: [
        { name: 'momentum', enabled: true },
        { name: 'mean_reversion', enabled: false }
      ]
    });
    
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByText(/AI Trading Bot/i)).toBeInTheDocument();
    });
  });

  test('handles API errors gracefully', async () => {
    api.getStatus.mockRejectedValue(new Error('Network error'));
    api.getPortfolio.mockRejectedValue(new Error('Network error'));
    api.getStrategies.mockRejectedValue(new Error('Network error'));
    
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByText(/Connection Error/i)).toBeInTheDocument();
    });
  });
});
