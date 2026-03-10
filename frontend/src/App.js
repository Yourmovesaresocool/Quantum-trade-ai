import React, { useState, useEffect, useCallback, useMemo, memo, createContext, useContext } from 'react';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import axios from 'axios';
import './animations.css';
import './darkmode.css';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';
const ML_SERVICE_URL = 'http://localhost:8000';

// ============================================
// DARK MODE CONTEXT (FIXED)
// ============================================
const DarkModeContext = createContext();

const DarkModeProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : true; // Default to dark mode
  });

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    // FIX: Apply correct class based on darkMode state
    document.body.className = darkMode ? 'dark-mode' : 'light-mode';
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(prev => !prev);

  return (
    <DarkModeContext.Provider value={{ darkMode, toggleDarkMode }}>
      {children}
    </DarkModeContext.Provider>
  );
};

const useDarkMode = () => {
  const context = useContext(DarkModeContext);
  if (!context) throw new Error('useDarkMode must be used within DarkModeProvider');
  return context;
};

const STOCK_DATABASE = [
  { symbol: 'AAPL', name: 'Apple Inc.', category: 'Tech', sector: 'Technology' },
  { symbol: 'MSFT', name: 'Microsoft Corp.', category: 'Tech', sector: 'Software' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', category: 'Tech', sector: 'Internet' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', category: 'Tech', sector: 'E-commerce' },
  { symbol: 'META', name: 'Meta Platforms', category: 'Tech', sector: 'Social Media' },
  { symbol: 'NVDA', name: 'NVIDIA Corp.', category: 'Tech', sector: 'AI/GPU' },
  { symbol: 'AMD', name: 'AMD Inc.', category: 'Tech', sector: 'Semiconductors' },
  { symbol: 'INTC', name: 'Intel Corp.', category: 'Tech', sector: 'Semiconductors' },
  { symbol: 'ORCL', name: 'Oracle Corp.', category: 'Tech', sector: 'Software' },
  { symbol: 'CRM', name: 'Salesforce', category: 'Tech', sector: 'Software' },
  { symbol: 'ADBE', name: 'Adobe Inc.', category: 'Tech', sector: 'Software' },
  { symbol: 'IBM', name: 'IBM', category: 'Tech', sector: 'Software' },
  { symbol: 'CSCO', name: 'Cisco Systems', category: 'Tech', sector: 'Networking' },
  { symbol: 'TSLA', name: 'Tesla Inc.', category: 'Auto', sector: 'EV' },
  { symbol: 'F', name: 'Ford Motor', category: 'Auto', sector: 'Automotive' },
  { symbol: 'GM', name: 'General Motors', category: 'Auto', sector: 'Automotive' },
  { symbol: 'JPM', name: 'JPMorgan Chase', category: 'Finance', sector: 'Banking' },
  { symbol: 'V', name: 'Visa Inc.', category: 'Finance', sector: 'Payments' },
  { symbol: 'MA', name: 'Mastercard', category: 'Finance', sector: 'Payments' },
  { symbol: 'BAC', name: 'Bank of America', category: 'Finance', sector: 'Banking' },
  { symbol: 'WFC', name: 'Wells Fargo', category: 'Finance', sector: 'Banking' },
  { symbol: 'GS', name: 'Goldman Sachs', category: 'Finance', sector: 'Banking' },
  { symbol: 'MS', name: 'Morgan Stanley', category: 'Finance', sector: 'Banking' },
  { symbol: 'PYPL', name: 'PayPal', category: 'Finance', sector: 'FinTech' },
  { symbol: 'WMT', name: 'Walmart Inc.', category: 'Retail', sector: 'Discount' },
  { symbol: 'TGT', name: 'Target Corp.', category: 'Retail', sector: 'Retail' },
  { symbol: 'COST', name: 'Costco', category: 'Retail', sector: 'Wholesale' },
  { symbol: 'HD', name: 'Home Depot', category: 'Retail', sector: 'Home Improvement' },
  { symbol: 'LOW', name: 'Lowes', category: 'Retail', sector: 'Home Improvement' },
  { symbol: 'NKE', name: 'Nike', category: 'Retail', sector: 'Apparel' },
  { symbol: 'SBUX', name: 'Starbucks', category: 'Retail', sector: 'Food & Beverage' },
  { symbol: 'MCD', name: 'McDonalds', category: 'Retail', sector: 'Food & Beverage' },
  { symbol: 'BTC-USD', name: 'Bitcoin', category: 'Crypto', sector: 'Cryptocurrency' },
  { symbol: 'ETH-USD', name: 'Ethereum', category: 'Crypto', sector: 'Cryptocurrency' },
  { symbol: 'DIS', name: 'Disney', category: 'Media', sector: 'Entertainment' },
  { symbol: 'NFLX', name: 'Netflix', category: 'Media', sector: 'Streaming' },
  { symbol: 'CMCSA', name: 'Comcast', category: 'Media', sector: 'Cable' },
  { symbol: 'BA', name: 'Boeing', category: 'Aerospace', sector: 'Defense' },
  { symbol: 'LMT', name: 'Lockheed Martin', category: 'Aerospace', sector: 'Defense' },
  { symbol: 'RTX', name: 'Raytheon', category: 'Aerospace', sector: 'Defense' },
  { symbol: 'JNJ', name: 'Johnson & Johnson', category: 'Healthcare', sector: 'Pharma' },
  { symbol: 'PFE', name: 'Pfizer', category: 'Healthcare', sector: 'Pharma' },
  { symbol: 'UNH', name: 'UnitedHealth', category: 'Healthcare', sector: 'Healthcare' },
  { symbol: 'ABBV', name: 'AbbVie', category: 'Healthcare', sector: 'Pharma' },
  { symbol: 'TMO', name: 'Thermo Fisher', category: 'Healthcare', sector: 'Healthcare' },
  { symbol: 'XOM', name: 'Exxon Mobil', category: 'Energy', sector: 'Oil' },
  { symbol: 'CVX', name: 'Chevron', category: 'Energy', sector: 'Oil' },
  { symbol: 'T', name: 'AT&T', category: 'Telecom', sector: 'Wireless' },
  { symbol: 'VZ', name: 'Verizon', category: 'Telecom', sector: 'Wireless' },
];

const TIME_RANGES = [
  { label: '1W', days: 7 },
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
  { label: '2Y', days: 730 },
  { label: '5Y', days: 1825 },
  { label: 'ALL', days: null }
];

const Candlestick = memo(({ data, width, height }) => {
  if (!data || data.length === 0) return null;
  const validData = data.filter(d => d.open && d.high && d.low && d.price);
  if (validData.length === 0) return null;
  const maxPrice = Math.max(...validData.map(d => d.high));
  const minPrice = Math.min(...validData.map(d => d.low));
  const priceRange = maxPrice - minPrice || 1;
  const padding = 20;
  const getY = (price) => padding + ((maxPrice - price) / priceRange) * (height - padding * 2);
  const barWidth = Math.max(2, Math.min(12, (width / validData.length) * 0.6));
  const barSpacing = width / validData.length;
  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      {validData.map((item, index) => {
        const x = index * barSpacing + barSpacing / 2;
        const isGreen = item.price >= item.open;
        const color = isGreen ? '#10B981' : '#EF4444';
        const highY = getY(item.high);
        const lowY = getY(item.low);
        const openY = getY(item.open);
        const closeY = getY(item.price);
        const bodyTop = Math.min(openY, closeY);
        const bodyHeight = Math.abs(closeY - openY);
        return (
          <g key={index}>
            <line x1={x} y1={highY} x2={x} y2={lowY} stroke={color} strokeWidth={1.5} />
            <rect x={x - barWidth / 2} y={bodyTop} width={barWidth} height={Math.max(bodyHeight, 1.5)} fill={color} stroke={color} strokeWidth={1} />
          </g>
        );
      })}
    </svg>
  );
});

const DarkModeToggle = () => {
  const { darkMode, toggleDarkMode } = useDarkMode();
  return (
    <button
      onClick={toggleDarkMode}
      className="p-3 rounded-xl bg-white/10 hover:bg-white/20 transition-all duration-300 transform hover:scale-110 flex items-center gap-2"
      title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      aria-label={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
    >
      <span className="text-2xl">{darkMode ? '☀️' : '🌙'}</span>
      <span className="hidden md:inline text-sm">{darkMode ? 'Light' : 'Dark'}</span>
    </button>
  );
};

const EmailAlertModal = memo(({ isOpen, onClose, symbol, signal }) => {
  const [email, setEmail] = useState(() => localStorage.getItem('userEmail') || '');
  const [sending, setSending] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSendAlert = async () => {
    if (!email || !email.includes('@')) {
      alert('❌ Please enter a valid email address');
      return;
    }
    
    setSending(true);
    try {
      localStorage.setItem('userEmail', email);
      
      await axios.post(`${API_URL}/api/send-alert`, {
        email,
        symbol,
        action: signal.decision.action,
        price: signal.current_price,
        reason: signal.decision.reason
      });
      
      setSuccess(true);
      setTimeout(() => {
        setSuccess(false);
        onClose();
      }, 2000);
    } catch (error) {
      alert('⚠️ Failed to send email. Make sure backend email service is configured.');
    } finally {
      setSending(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-gray-900 border-2 border-indigo-500 rounded-2xl p-8 max-w-md w-full shadow-2xl" onClick={e => e.stopPropagation()}>
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <span>📧</span> Email This Signal
        </h3>
        <div className="mb-6">
          <label className="text-sm text-gray-400 block mb-2">Your Email Address</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="your@email.com"
            className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-3 focus:border-indigo-500 focus:outline-none transition"
          />
        </div>
        <div className="bg-indigo-900/30 border border-indigo-500/30 rounded-xl p-4 mb-6">
          <p className="text-sm text-gray-300 mb-1">
            Signal: <strong className="text-white text-lg">{signal?.decision.action}</strong> {symbol}
          </p>
          <p className="text-sm text-gray-300">
            Price: <strong className="text-white text-lg">${signal?.current_price?.toFixed(2)}</strong>
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleSendAlert}
            disabled={sending || !email}
            className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed p-4 rounded-xl font-bold transition"
          >
            {sending ? '⏳ Sending...' : success ? '✅ Sent!' : '📧 Send Alert'}
          </button>
          <button
            onClick={onClose}
            className="px-6 bg-gray-700 hover:bg-gray-600 p-4 rounded-xl font-bold transition"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
});

const AddToWatchlistButton = memo(({ symbol, watchlist, toggleWatchlist }) => {
  const inWatchlist = watchlist.includes(symbol);
  const [justAdded, setJustAdded] = useState(false);

  const handleClick = () => {
    toggleWatchlist(symbol);
    if (!inWatchlist) {
      setJustAdded(true);
      setTimeout(() => setJustAdded(false), 2000);
    }
  };

  return (
    <button
      onClick={handleClick}
      className={`w-full px-4 py-3 rounded-xl font-bold transition-all duration-300 transform hover:scale-105 ${
        inWatchlist 
          ? 'bg-yellow-500/30 text-yellow-300 border-2 border-yellow-400 shadow-lg' 
          : 'bg-gray-700 text-gray-300 border-2 border-gray-600 hover:border-yellow-500 hover:bg-gray-600'
      }`}
    >
      {justAdded ? '✅ Added to Watchlist!' : inWatchlist ? '⭐ In Watchlist' : '➕ Add to Watchlist'}
    </button>
  );
});

const ModelStatusBadge = memo(({ modelInfo }) => {
  if (!modelInfo) return null;
  const isLSTM = modelInfo.type === 'LSTM';
  return (
    <div className={`px-4 py-2 rounded-lg ${isLSTM ? 'bg-green-500/20 border-green-500' : 'bg-yellow-500/20 border-yellow-500'} border`}>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${isLSTM ? 'bg-green-400' : 'bg-yellow-400'} animate-pulse`} />
        <span className="text-sm font-semibold">{isLSTM ? '🧠 LSTM Deep Learning' : '📊 Ensemble Model'}</span>
        {isLSTM && modelInfo.metadata?.r2_score && (
          <span className="text-xs text-green-400">({(modelInfo.metadata.r2_score * 100).toFixed(1)}% Accuracy)</span>
        )}
      </div>
    </div>
  );
});

const PriceCard = memo(({ symbol, currentPrice, priceChange, holdings }) => {
  const stock = STOCK_DATABASE.find(s => s.symbol === symbol);
  return (
    <div className="bg-gradient-to-br from-indigo-600/30 to-purple-600/30 backdrop-blur-xl border border-indigo-500/30 rounded-2xl p-6 shadow-2xl">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-3xl font-black">{symbol}</h2>
          <p className="text-sm text-gray-300">{stock?.name || 'Unknown'}</p>
          <p className="text-xs text-indigo-400 mt-1">{stock?.sector || ''}</p>
        </div>
      </div>
      <div className="mb-4">
        <p className="text-5xl font-black">${currentPrice.toFixed(2)}</p>
        <span className={`text-sm font-bold flex items-center gap-1 ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>{priceChange >= 0 ? '↗' : '↘'} {Math.abs(priceChange).toFixed(2)}%</span>
      </div>
      {holdings > 0 && (
        <div className="bg-black/40 p-3 rounded-lg border border-indigo-400/30 mb-4">
          <p className="text-xs text-gray-400">Your Holdings</p>
          <p className="text-xl font-bold text-indigo-400">{holdings.toFixed(4)} shares</p>
          <p className="text-xs text-gray-400">Value: ${(holdings * currentPrice).toFixed(2)}</p>
        </div>
      )}
    </div>
  );
});

const SignalCard = memo(({ signal, onViewReasoning, onEmailAlert }) => {
  if (!signal) return null;
  const colorClass = { BUY: 'bg-green-600/30 border-green-500/50', SELL: 'bg-red-600/30 border-red-500/50', HOLD: 'bg-yellow-600/30 border-yellow-500/50' };
  return (
    <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-2xl ${colorClass[signal.decision.action]}`}>
      <h3 className="text-xs font-bold uppercase mb-3 text-gray-200">AI Trading Signal</h3>
      <div className="text-center">
        <p className="text-5xl mb-2">{signal.decision.action === 'BUY' ? '📈' : signal.decision.action === 'SELL' ? '📉' : '⏸️'}</p>
        <p className="text-4xl font-black mb-3">{signal.decision.action}</p>
        <button onClick={onViewReasoning} className="w-full bg-white/20 hover:bg-white/30 backdrop-blur p-3 rounded-lg font-semibold transition mb-2">🧠 Why This Signal?</button>
        <button onClick={onEmailAlert} className="w-full bg-indigo-600/50 hover:bg-indigo-600/70 backdrop-blur p-3 rounded-lg font-semibold transition">📧 Email This Signal</button>
      </div>
    </div>
  );
});

const PredictionCard = memo(({ prediction }) => {
  if (!prediction) return null;
  return (
    <div className="bg-gray-900/90 backdrop-blur-xl border border-gray-800 rounded-2xl p-6 shadow-2xl">
      <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">🔮 ML Prediction</h3>
      <div className="text-center">
        <p className="text-xs text-gray-400 mb-2">Next Day Forecast</p>
        <p className="text-4xl font-black text-indigo-400 mb-3">${prediction.predicted_price?.toFixed(2) || 'N/A'}</p>
        {prediction.confidence && (
          <div className="mt-3 text-xs">
            <p className="text-gray-500">Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
            <p className="text-gray-600 text-[10px] mt-1">LSTM Deep Learning</p>
          </div>
        )}
      </div>
    </div>
  );
});

const TrendingStocks = memo(({ onSelectStock, currentSymbol, watchlist, toggleWatchlist }) => {
  const trending = ['NVDA', 'TSLA', 'AAPL', 'BTC-USD', 'GOOGL', 'AMZN'];
  return (
    <div className="bg-gray-900/90 backdrop-blur-xl border border-gray-800 rounded-2xl p-6">
      <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">🔥 Trending Now</h3>
      <div className="space-y-2">
        {trending.map(sym => {
          const stock = STOCK_DATABASE.find(s => s.symbol === sym);
          const isSelected = sym === currentSymbol;
          const inWatchlist = watchlist?.includes(sym);
          return (
            <div key={sym} className={`rounded-lg transition ${isSelected ? 'bg-indigo-600/50 border border-indigo-400' : 'bg-gray-800/50'}`}>
              <div className="flex items-center p-3">
                <button onClick={() => onSelectStock(sym)} className="flex-1 text-left">
                  <div className="flex justify-between items-center">
                    <div><p className="font-bold">{sym}</p><p className="text-xs text-gray-400">{stock?.name}</p></div>
                    <span className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded">HOT</span>
                  </div>
                </button>
                <button onClick={() => toggleWatchlist(sym)} className={`ml-2 p-2 rounded transition ${inWatchlist ? 'text-yellow-400' : 'text-gray-500 hover:text-yellow-400'}`}>⭐</button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
});

const CustomTooltip = memo(({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-gray-900/95 border border-indigo-500 p-4 rounded-xl shadow-2xl backdrop-blur">
        <p className="text-xs text-gray-400 mb-2 font-semibold">{data.fullDate}</p>
        {data.open !== undefined ? (
          <div className="text-xs space-y-1">
            <p className="text-white font-bold">Open: ${data.open.toFixed(2)}</p>
            <p className="text-green-400 font-bold">High: ${data.high.toFixed(2)}</p>
            <p className="text-red-400 font-bold">Low: ${data.low.toFixed(2)}</p>
            <p className="text-white font-bold">Close: ${data.price.toFixed(2)}</p>
            <p className={`font-bold ${data.price >= data.open ? 'text-green-400' : 'text-red-400'}`}>{data.price >= data.open ? '↗' : '↘'} ${Math.abs(data.price - data.open).toFixed(2)}</p>
          </div>
        ) : (
          <p className="text-white font-bold text-lg">{data.isPrediction && '🔮 '}${payload[0].value?.toFixed(2)}</p>
        )}
        {data.volume && <p className="text-xs text-gray-400 mt-1 pt-1 border-t border-gray-700">Vol: {(data.volume / 1000000).toFixed(2)}M</p>}
      </div>
    );
  }
  return null;
});

function MainApp() {
  const [userId] = useState(() => {
    let id = localStorage.getItem('userId');
    if (!id) {
      id = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('userId', id);
    }
    return id;
  });

  const getUserKey = useCallback((key) => `${userId}_${key}`, [userId]);
  
  const [symbol, setSymbol] = useState(() => localStorage.getItem(getUserKey('lastSymbol')) || 'AAPL');
  const [searchQuery, setSearchQuery] = useState('');
  const [prices, setPrices] = useState([]);
  const [pricesLoading, setPricesLoading] = useState(false);
  const [signal, setSignal] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  
  const [purchasePrices, setPurchasePrices] = useState(() => {
    const saved = localStorage.getItem(getUserKey('purchasePrices'));
    return saved ? JSON.parse(saved) : {};
  });
  
  const [portfolio, setPortfolio] = useState(() => {
    const saved = localStorage.getItem(getUserKey('portfolio'));
    const initialBalance = parseFloat(localStorage.getItem(getUserKey('initialBalance')) || '10000');
    return saved ? JSON.parse(saved) : { balance: initialBalance, holdings: {} };
  });
  
  const [watchlist, setWatchlist] = useState(() => {
    const saved = localStorage.getItem(getUserKey('watchlist'));
    return saved ? JSON.parse(saved) : ['AAPL', 'GOOGL', 'BTC-USD'];
  });
  
  const [showModal, setShowModal] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [chartType, setChartType] = useState('area');
  
  const [tradeHistory, setTradeHistory] = useState(() => {
    const saved = localStorage.getItem(getUserKey('tradeHistory'));
    return saved ? JSON.parse(saved) : [];
  });
  
  const [activeTab, setActiveTab] = useState('analysis');
  const [error, setError] = useState(null);
  
  const [stockPrices, setStockPrices] = useState(() => {
    const saved = localStorage.getItem(getUserKey('stockPrices'));
    return saved ? JSON.parse(saved) : {};
  });
  
  const [initialBalance] = useState(() => {
    const saved = localStorage.getItem(getUserKey('initialBalance'));
    if (saved) return parseFloat(saved);
    localStorage.setItem(getUserKey('initialBalance'), '10000');
    return 10000;
  });
  
  const [manualTradeAmount, setManualTradeAmount] = useState('');
  const [selectedTimeRange, setSelectedTimeRange] = useState('1Y');
  const [customStartDate, setCustomStartDate] = useState('');
  const [customEndDate, setCustomEndDate] = useState('');
  const [useCustomRange, setUseCustomRange] = useState(false);

  // Get dark mode from context
  const { darkMode } = useDarkMode();

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const res = await axios.get(`${ML_SERVICE_URL}/health`);
        setModelInfo(res.data.model);
      } catch (err) {
        console.warn('Could not fetch ML model info:', err.message);
      }
    };
    fetchModelInfo();
  }, []);

  useEffect(() => { localStorage.setItem(getUserKey('lastSymbol'), symbol); }, [symbol, getUserKey]);
  useEffect(() => { 
    const timer = setTimeout(() => { localStorage.setItem(getUserKey('portfolio'), JSON.stringify(portfolio)); }, 500);
    return () => clearTimeout(timer);
  }, [portfolio, getUserKey]);
  useEffect(() => { localStorage.setItem(getUserKey('watchlist'), JSON.stringify(watchlist)); }, [watchlist, getUserKey]);
  useEffect(() => { 
    const timer = setTimeout(() => { localStorage.setItem(getUserKey('tradeHistory'), JSON.stringify(tradeHistory)); }, 500);
    return () => clearTimeout(timer);
  }, [tradeHistory, getUserKey]);
  useEffect(() => { localStorage.setItem(getUserKey('stockPrices'), JSON.stringify(stockPrices)); }, [stockPrices, getUserKey]);
  useEffect(() => { localStorage.setItem(getUserKey('purchasePrices'), JSON.stringify(purchasePrices)); }, [purchasePrices, getUserKey]);

  const filteredStocks = useMemo(() => 
    STOCK_DATABASE.filter(s => 
      s.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.category.toLowerCase().includes(searchQuery.toLowerCase())
    ), [searchQuery]
  );

  const loadPrices = useCallback(async () => {
    try {
      setPricesLoading(true);
      setError(null);
      const res = await axios.get(`${API_URL}/api/prices/${symbol}?limit=2000`);
      if (res.data?.success) {
        const priceData = res.data.data.reverse();
        setPrices(priceData);
        if (priceData.length > 0) {
          const currentPrice = parseFloat(priceData[priceData.length - 1].close);
          setStockPrices(prev => ({ ...prev, [symbol]: currentPrice }));
        }
      }
    } catch (e) {
      console.error('Price fetch error:', e);
      setError(`Failed to load ${symbol} data.`);
      setPrices([]);
    } finally {
      setPricesLoading(false);
    }
  }, [symbol]);

  useEffect(() => { loadPrices(); }, [loadPrices]);

  const currentPrice = useMemo(() => prices.length > 0 ? parseFloat(prices[prices.length - 1].close) : 0, [prices]);
  const priceChange = useMemo(() => prices.length > 1 ? ((currentPrice - parseFloat(prices[prices.length - 2].close)) / parseFloat(prices[prices.length - 2].close) * 100) : 0, [prices, currentPrice]);

  const filteredChartData = useMemo(() => {
    if (!prices.length) return [];
    let filteredPrices = [...prices];
    if (useCustomRange && customStartDate && customEndDate) {
      const start = new Date(customStartDate);
      const end = new Date(customEndDate);
      filteredPrices = prices.filter(p => {
        const date = new Date(p.timestamp);
        return date >= start && date <= end;
      });
    } else {
      const range = TIME_RANGES.find(r => r.label === selectedTimeRange);
      if (range && range.days) {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - range.days);
        filteredPrices = prices.filter(p => new Date(p.timestamp) >= cutoffDate);
      }
    }
    const data = filteredPrices.map((p) => {
      const date = new Date(p.timestamp);
      return {
        date: date.getDate().toString(),
        fullDate: date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
        price: parseFloat(p.close),
        volume: parseInt(p.volume),
        high: parseFloat(p.high),
        low: parseFloat(p.low),
        open: parseFloat(p.open),
      };
    });
    if (prediction && data.length > 0) {
      const lastDate = new Date(filteredPrices[filteredPrices.length - 1].timestamp);
      lastDate.setDate(lastDate.getDate() + 1);
      data.push({ date: 'F', fullDate: 'AI Prediction - ' + lastDate.toLocaleDateString('en-US'), price: prediction.predicted_price, isPrediction: true });
    }
    return data;
  }, [prices, prediction, selectedTimeRange, customStartDate, customEndDate, useCustomRange]);

  const availableYears = useMemo(() => {
    if (!prices.length) return [];
    const years = new Set(prices.map(p => new Date(p.timestamp).getFullYear()));
    return Array.from(years).sort((a, b) => b - a);
  }, [prices]);

  const generateSimpleReasoning = useCallback((stockSymbol, action) => {
    const stock = STOCK_DATABASE.find(s => s.symbol === stockSymbol);
    const stockName = stock?.name || stockSymbol;
    const actionEmoji = action === 'BUY' ? '📈' : action === 'SELL' ? '📉' : '⏸️';
    
    const simpleExplanations = {
      'BUY': [
        `${actionEmoji} **Why BUY ${stockName}?**\n\nThe AI detected that ${stockName} stock price is trending UPWARD. This means the price has been going up recently. When prices go up, it's usually a good time to consider buying.\n\n**What this means for you:** If you buy now, you might make money if the price keeps going up.`,
        `${actionEmoji} **Good News for ${stockName}!**\n\nOur AI found strong POSITIVE signals. Think of it like this: the stock is gaining momentum, like a car speeding up on the highway.\n\n**Simple explanation:** More people want to buy this stock, which pushes the price higher. That's a good sign!`
      ],
      'SELL': [
        `${actionEmoji} **Why SELL ${stockName}?**\n\nThe AI detected ${stockName} is trending DOWNWARD. The price has been dropping. When prices fall, it might be smart to sell before losing more money.\n\n**What this means for you:** If you own this stock, selling now could protect your money from bigger losses.`,
        `${actionEmoji} **Caution on ${stockName}**\n\nOur AI found warning signals. Think of it as a yellow traffic light - time to slow down and be careful.\n\n**Simple explanation:** The stock is losing value. Selling means you can use that money to buy something better.`
      ],
      'HOLD': [
        `${actionEmoji} **HOLD ${stockName} for Now**\n\nThe AI sees ${stockName} in a neutral zone - not going up or down strongly. It's like a car at a red light, waiting for the signal to change.\n\n**What this means for you:** Don't buy or sell yet. Wait and watch for clearer signals.`
      ]
    };
    
    const explanations = simpleExplanations[action] || simpleExplanations['HOLD'];
    const randomIndex = Math.floor(Math.random() * explanations.length);
    return explanations[randomIndex];
  }, []);

  const handleManualBuy = useCallback(() => {
    const shares = parseFloat(manualTradeAmount);
    if (!manualTradeAmount || isNaN(shares) || shares <= 0) { setError('❌ Enter valid shares'); return; }
    const cost = shares * currentPrice;
    if (cost > portfolio.balance) { setError(`❌ Insufficient funds`); return; }
    
    setPortfolio(prev => ({ 
      balance: prev.balance - cost, 
      holdings: { ...prev.holdings, [symbol]: (prev.holdings[symbol] || 0) + shares } 
    }));
    
    setPurchasePrices(prev => {
      const existingShares = portfolio.holdings[symbol] || 0;
      const existingAvgPrice = prev[symbol] || currentPrice;
      const newAvgPrice = ((existingAvgPrice * existingShares) + (currentPrice * shares)) / (existingShares + shares);
      return { ...prev, [symbol]: newAvgPrice };
    });
    
    setTradeHistory(prev => [{
      action:'BUY',
      symbol,
      price:currentPrice,
      shares,
      time:new Date().toLocaleTimeString(),
      date:new Date().toLocaleDateString(),
      id:Date.now(),
      reason:`Manual: ${shares} @ $${currentPrice.toFixed(2)}`
    },...prev.slice(0,99)]);
    setManualTradeAmount('');
    setError(null);
  }, [manualTradeAmount, currentPrice, portfolio, symbol]);

  const handleManualSell = useCallback(() => {
    const shares = parseFloat(manualTradeAmount);
    const currentShares = portfolio.holdings[symbol] || 0;
    if (!manualTradeAmount || isNaN(shares) || shares <= 0 || shares > currentShares) { 
      setError('❌ Invalid shares'); 
      return; 
    }
    
    const revenue = shares * currentPrice;
    const avgPurchasePrice = purchasePrices[symbol] || currentPrice;
    const profitLoss = (currentPrice - avgPurchasePrice) * shares;
    
    setPortfolio(prev => ({ 
      balance: prev.balance + revenue, 
      holdings: { ...prev.holdings, [symbol]: Math.max(0, prev.holdings[symbol] - shares) } 
    }));
    
    setTradeHistory(prev => [{
      action:'SELL',
      symbol,
      price:currentPrice,
      shares,
      profitLoss,
      time:new Date().toLocaleTimeString(),
      date:new Date().toLocaleDateString(),
      id:Date.now(),
      reason:`Sold ${shares} @ $${currentPrice.toFixed(2)} (P/L: ${profitLoss>=0?'+':''}$${profitLoss.toFixed(2)})`
    },...prev.slice(0,99)]);
    setManualTradeAmount('');
    setError(null);
  }, [manualTradeAmount, currentPrice, portfolio, symbol, purchasePrices]);

  const exportTradeHistory = useCallback(() => {
    if (tradeHistory.length === 0) return;
    const csv = ['Date,Time,Symbol,Action,Price,Shares,P/L,Reason', ...tradeHistory.map(t => `${t.date},${t.time},${t.symbol},${t.action},${t.price?.toFixed(2)||0},${t.shares?.toFixed(4)||0},${t.profitLoss?.toFixed(2)||0},"${t.reason||''}"`)].join('\n');
    const blob = new Blob([csv], {type:'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trades-${symbol}-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [tradeHistory, symbol]);

  const handleAnalyze = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [predRes, tradeRes] = await Promise.all([
        axios.post(`${API_URL}/api/predict`, { symbol }).catch(() => ({data:{success:false}})),
        axios.post(`${API_URL}/api/trade`, { symbol, balance: portfolio.balance, shares: portfolio.holdings[symbol]||0 }).catch(() => ({data:{success:false}}))
      ]);
      if (predRes.data?.success) setPrediction(predRes.data.prediction); else setPrediction(null);
      if (tradeRes.data?.success) {
        const enhancedSignal = { ...tradeRes.data, decision: { ...tradeRes.data.decision, reason: generateSimpleReasoning(symbol, tradeRes.data.decision.action) } };
        setSignal(enhancedSignal);
        setTradeHistory(prev => [{ ...enhancedSignal.decision, symbol, price: tradeRes.data.current_price, time: new Date().toLocaleTimeString(), date: new Date().toLocaleDateString(), id: Date.now() }, ...prev.slice(0, 99)]);
      } else setSignal(null);
    } catch (error) {
      setError('AI analysis failed');
    } finally {
      setLoading(false);
    }
  }, [symbol, portfolio, generateSimpleReasoning]);

  const toggleWatchlist = useCallback((sym) => { 
    setWatchlist(prev => prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]); 
  }, []);

  const totalPortfolioValue = useMemo(() => {
    const holdingsValue = Object.entries(portfolio.holdings).reduce((sum, [sym, shares]) => {
      const stockPrice = sym === symbol ? currentPrice : (stockPrices[sym] || 0);
      return sum + (shares * stockPrice);
    }, 0);
    return portfolio.balance + holdingsValue;
  }, [portfolio, currentPrice, symbol, stockPrices]);

  const profitLoss = useMemo(() => totalPortfolioValue - initialBalance, [totalPortfolioValue, initialBalance]);

  const renderChart = useCallback(() => {
    const commonProps = { data: filteredChartData, margin: { top: 10, right: 30, left: 0, bottom: 0 } };
    const xAxisProps = { dataKey: "date", stroke: "#6B7280", tick: { fill: '#9CA3AF', fontSize: 11 }, angle: -45, textAnchor: 'end', height: 60 };
    const yAxisProps = { stroke: "#6B7280", tick: { fill: '#9CA3AF', fontSize: 11 }, width: 80, tickFormatter: (value) => `$${value.toFixed(0)}` };
    const sharedComponents = (<><CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} /><XAxis {...xAxisProps} /><YAxis {...yAxisProps} /><Tooltip content={<CustomTooltip />} /></>);
    if (chartType === 'candlestick') {
      return (<ResponsiveContainer width="100%" height={400}>{({ width, height }) => (<ComposedChart {...commonProps} width={width} height={height}>{sharedComponents}<Candlestick data={filteredChartData} width={width} height={height - 80} /></ComposedChart>)}</ResponsiveContainer>);
    } else if (chartType === 'area') {
      return (<ResponsiveContainer width="100%" height={400}><AreaChart {...commonProps}><defs><linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#6366F1" stopOpacity={0.8}/><stop offset="95%" stopColor="#6366F1" stopOpacity={0}/></linearGradient></defs>{sharedComponents}<Area type="monotone" dataKey="price" stroke="#6366F1" strokeWidth={2} fillOpacity={1} fill="url(#colorPrice)" /></AreaChart></ResponsiveContainer>);
    } else if (chartType === 'line') {
      return (<ResponsiveContainer width="100%" height={400}><LineChart {...commonProps}>{sharedComponents}<Line type="monotone" dataKey="price" stroke="#6366F1" strokeWidth={2} dot={false} activeDot={{ r: 5 }} /></LineChart></ResponsiveContainer>);
    } else {
      return (<ResponsiveContainer width="100%" height={400}><BarChart {...commonProps}>{sharedComponents}<Bar dataKey="price" fill="#6366F1" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer>);
    }
  }, [chartType, filteredChartData]);

  return (
    <div className={`min-h-screen relative overflow-hidden transition-colors duration-300 ${
      darkMode 
        ? 'bg-gradient-to-br from-gray-950 via-slate-900 to-gray-950 text-white' 
        : 'bg-gradient-to-br from-gray-50 via-blue-50 to-gray-100 text-gray-900'
    }`}>
      <div className="fixed inset-0 opacity-5 pointer-events-none bg-[radial-gradient(circle_at_50%_50%,rgba(99,102,241,0.1),transparent_50%)]" />
      
      <nav className={`relative z-10 shadow-2xl sticky top-0 backdrop-blur transition-colors duration-300 ${
        darkMode 
          ? 'bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600' 
          : 'bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500'
      }`}>
        <div className="max-w-[1920px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-6">
              <h1 className="text-2xl md:text-3xl font-black tracking-tight flex items-center gap-2">
                <span className="text-3xl md:text-4xl">⚡</span>QUANTUM TRADE AI
              </h1>
              <div className="hidden md:flex items-center gap-2"><ModelStatusBadge modelInfo={modelInfo} /></div>
            </div>
            <div className="flex items-center gap-4 md:gap-6">
              <DarkModeToggle />
              <div className="text-right">
                <p className="text-xs opacity-75">Portfolio</p>
                <p className="text-lg md:text-xl font-bold text-green-300">
                  ${totalPortfolioValue.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs opacity-75">Cash</p>
                <p className="text-lg md:text-xl font-bold text-indigo-300">
                  ${portfolio.balance.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs opacity-75">P/L</p>
                <p className={`text-lg md:text-xl font-bold ${profitLoss>=0?'text-green-300':'text-red-300'}`}>
                  {profitLoss>=0?'+':''}${profitLoss.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
                </p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {error && (
        <div className="relative z-10 max-w-[1920px] mx-auto px-6 pt-4">
          <div className="bg-red-500/20 border border-red-500 rounded-xl p-4 flex justify-between items-center backdrop-blur animate-pulse">
            <p className="text-red-300">⚠️ {error}</p>
            <button onClick={() => setError(null)} className="text-red-300 hover:text-white text-xl">✕</button>
          </div>
        </div>
      )}

      <div className="relative z-10 max-w-[1920px] mx-auto px-6 pt-6">
        <div className="flex gap-2 md:gap-3 mb-6 overflow-x-auto pb-2">
          {[
            {id:'analysis',label:'📊 Analysis',icon:'📊'},
            {id:'watchlist',label:'⭐ Watchlist',icon:'⭐'},
            {id:'portfolio',label:'💼 Portfolio',icon:'💼'},
            {id:'history',label:'📜 History',icon:'📜'},
            {id:'about',label:'ℹ️ About',icon:'ℹ️'}
          ].map(tab => (
            <button 
              key={tab.id} 
              onClick={() => setActiveTab(tab.id)} 
              className={`px-4 md:px-6 py-2 md:py-3 rounded-xl font-semibold transition whitespace-nowrap text-sm md:text-base ${
                activeTab===tab.id
                  ?'bg-indigo-600 shadow-lg scale-105'
                  :'bg-gray-800/50 hover:bg-gray-700/50'
              }`}
            >
              <span className="md:hidden">{tab.icon}</span>
              <span className="hidden md:inline">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      <main className="relative z-10 max-w-[1920px] mx-auto px-6 pb-12">
        {activeTab==='analysis' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-3 space-y-6">
              <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
                darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
              }`}>
                <h3 className={`text-sm font-bold uppercase mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  🔍 Search Stocks
                </h3>
                <input 
                  type="text" 
                  value={searchQuery} 
                  onChange={(e) => setSearchQuery(e.target.value)} 
                  placeholder="Search symbol, name..." 
                  className={`w-full border rounded-xl px-4 py-3 text-sm focus:border-indigo-500 focus:outline-none transition ${
                    darkMode 
                      ? 'bg-gray-800 border-gray-700 text-white' 
                      : 'bg-gray-50 border-gray-300 text-gray-900'
                  }`}
                />
                {searchQuery && filteredStocks.length > 0 && (
                  <div className="mt-2 max-h-64 overflow-y-auto space-y-1">
                    {filteredStocks.slice(0, 20).map(stock => (
                      <div 
                        key={stock.symbol} 
                        className={`p-3 rounded-lg transition ${
                          stock.symbol===symbol
                            ?'bg-indigo-600/50'
                            : darkMode 
                              ? 'bg-gray-800 hover:bg-gray-700' 
                              : 'bg-gray-100 hover:bg-gray-200'
                        }`}
                      >
                        <div className="flex justify-between items-center">
                          <button 
                            onClick={() => { 
                              setSymbol(stock.symbol); 
                              setSearchQuery(''); 
                              setSignal(null); 
                              setPrediction(null); 
                            }} 
                            className="flex-1 text-left"
                          >
                            <p className="font-bold text-sm">{stock.symbol}</p>
                            <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                              {stock.name}
                            </p>
                          </button>
                          <div className="flex items-center gap-2">
                            <span className="text-xs px-2 py-1 bg-indigo-500/20 text-indigo-400 rounded">
                              {stock.category}
                            </span>
                            <button 
                              onClick={() => toggleWatchlist(stock.symbol)} 
                              className={`p-1.5 rounded transition ${
                                watchlist.includes(stock.symbol)
                                  ?'text-yellow-400'
                                  :'text-gray-500 hover:text-yellow-400'
                              }`}
                            >
                              ⭐
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {searchQuery && filteredStocks.length === 0 && (
                  <p className={`text-xs mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                    No stocks found
                  </p>
                )}
              </div>
              
              <PriceCard symbol={symbol} currentPrice={currentPrice} priceChange={priceChange} holdings={portfolio.holdings[symbol]||0} />
              <AddToWatchlistButton symbol={symbol} watchlist={watchlist} toggleWatchlist={toggleWatchlist} />
              
              <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
                darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
              }`}>
                <h3 className={`text-sm font-bold uppercase mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  💰 Manual Trading
                </h3>
                <div className="mb-4">
                  <label className={`text-xs mb-2 block ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Shares to Trade
                  </label>
                  <input 
                    type="number" 
                    step="0.0001" 
                    value={manualTradeAmount} 
                    onChange={(e) => setManualTradeAmount(e.target.value)} 
                    placeholder="Enter shares..." 
                    className={`w-full border rounded-xl px-4 py-3 text-sm focus:border-indigo-500 focus:outline-none transition ${
                      darkMode 
                        ? 'bg-gray-800 border-gray-700 text-white' 
                        : 'bg-gray-50 border-gray-300 text-gray-900'
                    }`}
                    min="0" 
                  />
                  {manualTradeAmount && !isNaN(parseFloat(manualTradeAmount)) && (
                    <p className={`text-xs mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Cost: ${(parseFloat(manualTradeAmount) * currentPrice).toFixed(2)}
                    </p>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <button 
                    onClick={handleManualBuy} 
                    disabled={!manualTradeAmount || currentPrice===0} 
                    className="bg-green-600 hover:bg-green-700 disabled:opacity-50 p-3 rounded-lg font-bold transition"
                  >
                    🛒 BUY
                  </button>
                  <button 
                    onClick={handleManualSell} 
                    disabled={!manualTradeAmount || currentPrice===0} 
                    className="bg-red-600 hover:bg-red-700 disabled:opacity-50 p-3 rounded-lg font-bold transition"
                  >
                    💵 SELL
                  </button>
                </div>
                <p className={`text-xs mt-3 text-center ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                  Holdings: {(portfolio.holdings[symbol]||0).toFixed(4)} shares
                </p>
              </div>
              
              <TrendingStocks 
                onSelectStock={(sym) => { setSymbol(sym); setSignal(null); setPrediction(null); }} 
                currentSymbol={symbol} 
                watchlist={watchlist} 
                toggleWatchlist={toggleWatchlist} 
              />
              
              <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
                darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
              }`}>
                <h3 className={`text-sm font-bold uppercase mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  🤖 AI Analysis
                </h3>
                <button 
                  onClick={handleAnalyze} 
                  disabled={loading||currentPrice===0} 
                  className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 p-4 rounded-xl font-bold disabled:opacity-50 transition shadow-lg text-white"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                      </svg>
                      Analyzing...
                    </span>
                  ) : (
                    '🧠 Run AI Analysis'
                  )}
                </button>
              </div>
              
              <SignalCard signal={signal} onViewReasoning={() => setShowModal(true)} onEmailAlert={() => setShowEmailModal(true)} />
              <PredictionCard prediction={prediction} />
            </div>

            <div className="lg:col-span-9 space-y-6">
              <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
                darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
              }`}>
                <h3 className={`text-sm font-bold uppercase mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  📅 Time Range
                </h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  {TIME_RANGES.map(range => (
                    <button 
                      key={range.label} 
                      onClick={() => { setSelectedTimeRange(range.label); setUseCustomRange(false); }} 
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
                        selectedTimeRange===range.label&&!useCustomRange
                          ?'bg-indigo-600 shadow-lg text-white'
                          : darkMode
                            ? 'bg-gray-800 hover:bg-gray-700 text-gray-200'
                            : 'bg-gray-100 hover:bg-gray-200 text-gray-900'
                      }`}
                    >
                      {range.label}
                    </button>
                  ))}
                </div>
                <div className={`border-t pt-4 ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                  <div className="flex items-center gap-2 mb-3">
                    <input 
                      type="checkbox" 
                      id="customRange" 
                      checked={useCustomRange} 
                      onChange={(e) => setUseCustomRange(e.target.checked)} 
                      className="w-4 h-4"
                    />
                    <label htmlFor="customRange" className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      Custom Date Range
                    </label>
                  </div>
                  {useCustomRange && (
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className={`text-xs mb-1 block ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          Start Date
                        </label>
                        <input 
                          type="date" 
                          value={customStartDate} 
                          onChange={(e) => setCustomStartDate(e.target.value)} 
                          className={`w-full border rounded-lg px-3 py-2 text-sm ${
                            darkMode 
                              ? 'bg-gray-800 border-gray-700 text-white' 
                              : 'bg-gray-50 border-gray-300 text-gray-900'
                          }`}
                        />
                      </div>
                      <div>
                        <label className={`text-xs mb-1 block ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          End Date
                        </label>
                        <input 
                          type="date" 
                          value={customEndDate} 
                          onChange={(e) => setCustomEndDate(e.target.value)} 
                          className={`w-full border rounded-lg px-3 py-2 text-sm ${
                            darkMode 
                              ? 'bg-gray-800 border-gray-700 text-white' 
                              : 'bg-gray-50 border-gray-300 text-gray-900'
                          }`}
                        />
                      </div>
                    </div>
                  )}
                </div>
                <div className={`mt-4 flex items-center justify-between text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  <span>📊 {filteredChartData.length} data points</span>
                  {availableYears.length > 0 && (
                    <span>Data: {availableYears[availableYears.length - 1]} - {availableYears[0]}</span>
                  )}
                </div>
              </div>

              <div className={`flex gap-4 items-center backdrop-blur-xl border rounded-2xl p-4 shadow-xl flex-wrap transition-colors duration-300 ${
                darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
              }`}>
                <p className={`text-sm font-semibold ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Chart Type:
                </p>
                <div className="flex gap-2">
                  {[
                    {type:'area',icon:'📈',label:'Area'},
                    {type:'line',icon:'📉',label:'Line'},
                    {type:'bar',icon:'📊',label:'Bar'},
                    {type:'candlestick',icon:'🕯️',label:'Candle'}
                  ].map(({type,icon,label}) => (
                    <button 
                      key={type} 
                      onClick={() => setChartType(type)} 
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
                        chartType===type
                          ?'bg-indigo-600 shadow-lg text-white'
                          : darkMode
                            ? 'bg-gray-800 hover:bg-gray-700 text-gray-200'
                            : 'bg-gray-100 hover:bg-gray-200 text-gray-900'
                      }`}
                    >
                      {icon} <span className="hidden sm:inline">{label}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
                darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
              }`}>
                <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
                  <h2 className="text-xl md:text-2xl font-bold flex items-center gap-2">
                    <span>💹</span>{symbol} Price Chart
                  </h2>
                  {prices.length > 0 && (
                    <div className="text-right">
                      <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Latest Price</p>
                      <p className="text-2xl font-bold">${currentPrice.toFixed(2)}</p>
                    </div>
                  )}
                </div>
                {pricesLoading ? (
                  <div className="flex items-center justify-center h-96">
                    <div className="text-center">
                      <div className="animate-spin h-12 w-12 border-4 border-indigo-500 border-t-transparent rounded-full mx-auto mb-4"></div>
                      <p>Loading {symbol} data...</p>
                    </div>
                  </div>
                ) : prices.length === 0 ? (
                  <div className="flex items-center justify-center h-96">
                    <div className={`text-center ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      <p className="text-6xl mb-4">📉</p>
                      <p className="text-xl">No price data for {symbol}</p>
                    </div>
                  </div>
                ) : (
                  renderChart()
                )}
              </div>

              {prices.length > 0 && (
                <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
                  darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
                }`}>
                  <h2 className="text-xl font-bold mb-6">📊 Trading Volume</h2>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={filteredChartData} margin={{top:10,right:30,left:0,bottom:0}}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3}/>
                      <XAxis dataKey="date" stroke="#6B7280" tick={{fill:'#9CA3AF',fontSize:11}} angle={-45} textAnchor="end" height={60}/>
                      <YAxis stroke="#6B7280" tick={{fill:'#9CA3AF',fontSize:11}} tickFormatter={(value) => `${(value/1000000).toFixed(0)}M`}/>
                      <Tooltip content={<CustomTooltip/>}/>
                      <Bar dataKey="volume" fill="#6366F1" radius={[4,4,0,0]}/>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab==='watchlist' && (
          <div>
            <div className="mb-6">
              <h2 className="text-3xl font-black mb-2">⭐ Your Watchlist</h2>
              <p className={darkMode ? 'text-gray-400' : 'text-gray-600'}>Quick access to favorites</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {watchlist.map(sym => {
                const stock = STOCK_DATABASE.find(s => s.symbol === sym);
                const stockPrice = stockPrices[sym] || 0;
                const holdings = portfolio.holdings[sym] || 0;
                return (
                  <div 
                    key={sym} 
                    className={`backdrop-blur-xl border rounded-2xl p-6 hover:border-indigo-500/50 transition shadow-xl ${
                      darkMode 
                        ? 'bg-gradient-to-br from-gray-900/90 to-gray-800/90 border-gray-700' 
                        : 'bg-gradient-to-br from-white/90 to-gray-100/90 border-gray-200'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-2xl font-bold">{sym}</h3>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {stock?.name||'Unknown'}
                        </p>
                        <p className="text-xs text-indigo-400">{stock?.sector}</p>
                      </div>
                      <button 
                        onClick={() => toggleWatchlist(sym)} 
                        className="text-red-400 hover:text-red-300 transition text-xl"
                      >
                        ✕
                      </button>
                    </div>
                    {stockPrice > 0 && (
                      <div className="mb-3">
                        <p className="text-2xl font-bold">${stockPrice.toFixed(2)}</p>
                      </div>
                    )}
                    {holdings > 0 && (
                      <div className="bg-black/30 p-2 rounded-lg mb-3">
                        <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Holdings</p>
                        <p className="text-sm font-bold text-indigo-400">{holdings.toFixed(4)} shares</p>
                      </div>
                    )}
                    <button 
                      onClick={() => { setSymbol(sym); setActiveTab('analysis'); setSignal(null); setPrediction(null); }} 
                      className="w-full bg-indigo-600 hover:bg-indigo-700 p-3 rounded-lg font-semibold transition text-white"
                    >
                      📊 Analyze
                    </button>
                  </div>
                );
              })}
              {watchlist.length === 0 && (
                <div className={`col-span-full text-center py-20 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  <p className="text-6xl mb-4">⭐</p>
                  <p className="text-xl">No stocks in watchlist</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab==='portfolio' && (
          <div className="space-y-6">
            <div className="mb-6">
              <h2 className="text-3xl font-black mb-2">💼 Portfolio Overview</h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-gradient-to-br from-green-600/30 to-green-800/30 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6 shadow-xl">
                <h3 className="text-sm text-gray-300 mb-2 font-semibold">💵 Cash Balance</h3>
                <p className="text-3xl md:text-4xl font-black text-green-300">
                  ${portfolio.balance.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
                </p>
              </div>
              <div className="bg-gradient-to-br from-indigo-600/30 to-indigo-800/30 backdrop-blur-xl border border-indigo-500/30 rounded-2xl p-6 shadow-xl">
                <h3 className="text-sm text-gray-300 mb-2 font-semibold">💎 Total Value</h3>
                <p className="text-3xl md:text-4xl font-black text-indigo-300">
                  ${totalPortfolioValue.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
                </p>
              </div>
              <div className={`bg-gradient-to-br backdrop-blur-xl border rounded-2xl p-6 shadow-xl ${
                profitLoss>=0
                  ?'from-green-600/30 to-green-800/30 border-green-500/30'
                  :'from-red-600/30 to-red-800/30 border-red-500/30'
              }`}>
                <h3 className="text-sm text-gray-300 mb-2 font-semibold">📈 Profit/Loss</h3>
                <p className={`text-3xl md:text-4xl font-black ${profitLoss>=0?'text-green-300':'text-red-300'}`}>
                  {profitLoss>=0?'+':''}${profitLoss.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
                </p>
              </div>
              <div className={`bg-gradient-to-br backdrop-blur-xl border rounded-2xl p-6 shadow-xl ${
                profitLoss>=0
                  ?'from-green-600/30 to-green-800/30 border-green-500/30'
                  :'from-red-600/30 to-red-800/30 border-red-500/30'
              }`}>
                <h3 className="text-sm text-gray-300 mb-2 font-semibold">📊 Return %</h3>
                <p className={`text-3xl md:text-4xl font-black ${profitLoss>=0?'text-green-300':'text-red-300'}`}>
                  {initialBalance>0?((profitLoss/initialBalance)*100).toFixed(2):'0.00'}%
                </p>
              </div>
            </div>

            <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
              darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
            }`}>
              <h2 className="text-2xl font-bold mb-6">📊 Current Holdings</h2>
              <div className="space-y-3">
                {Object.entries(portfolio.holdings).filter(([_,shares])=>shares>0).map(([sym,shares]) => {
                  const stock = STOCK_DATABASE.find(s => s.symbol === sym);
                  const stockPrice = sym===symbol?currentPrice:(stockPrices[sym]||0);
                  const holdingValue = shares*stockPrice;
                  const avgPurchasePrice = purchasePrices[sym]||stockPrice;
                  const unrealizedPL = (stockPrice-avgPurchasePrice)*shares;
                  return (
                    <div 
                      key={sym} 
                      className={`p-4 rounded-xl flex justify-between items-center transition ${
                        darkMode 
                          ? 'bg-gradient-to-r from-gray-800/50 to-indigo-900/30 hover:from-gray-700/50 hover:to-indigo-800/30' 
                          : 'bg-gradient-to-r from-gray-100/50 to-indigo-100/30 hover:from-gray-200/50 hover:to-indigo-200/30'
                      }`}
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="font-bold text-xl">{sym}</p>
                          <span className="text-xs px-2 py-1 bg-indigo-500/20 text-indigo-400 rounded">
                            {stock?.category}
                          </span>
                        </div>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {stock?.name}
                        </p>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <p className={darkMode ? 'text-gray-500' : 'text-gray-600'}>Shares</p>
                            <p className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                              {shares.toFixed(4)}
                            </p>
                          </div>
                          <div>
                            <p className={darkMode ? 'text-gray-500' : 'text-gray-600'}>Value</p>
                            <p className="text-indigo-400 font-semibold">${holdingValue.toFixed(2)}</p>
                          </div>
                        </div>
                        <p className={`text-sm font-bold mt-2 ${unrealizedPL>=0?'text-green-400':'text-red-400'}`}>
                          P/L: {unrealizedPL>=0?'+':''}${unrealizedPL.toFixed(2)}
                        </p>
                      </div>
                      <button 
                        onClick={() => { setSymbol(sym); setActiveTab('analysis'); }} 
                        className="ml-4 px-6 py-3 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-sm font-semibold transition text-white"
                      >
                        📊 Analyze
                      </button>
                    </div>
                  );
                })}
                {Object.values(portfolio.holdings).every(v=>v===0) && (
                  <div className={`text-center py-20 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                    <p className="text-6xl mb-4">📂</p>
                    <p className="text-xl">No holdings yet</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab==='history' && (
          <div>
            <div className="flex justify-between items-center mb-6 flex-wrap gap-4">
              <div>
                <h2 className="text-3xl font-black mb-2">📜 Trading History</h2>
              </div>
              <button 
                onClick={exportTradeHistory} 
                disabled={tradeHistory.length===0} 
                className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-6 py-3 rounded-lg font-semibold transition text-white"
              >
                📥 Export CSV
              </button>
            </div>
            <div className={`backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-colors duration-300 ${
              darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
            }`}>
              <div className="space-y-3 max-h-[700px] overflow-y-auto">
                {tradeHistory.length > 0 ? tradeHistory.map((trade) => (
                  <div 
                    key={trade.id} 
                    className={`p-5 rounded-xl transition shadow-lg ${
                      darkMode 
                        ? 'bg-gradient-to-r from-gray-800/70 to-gray-700/70 hover:from-gray-700/70 hover:to-gray-600/70' 
                        : 'bg-gradient-to-r from-gray-100/70 to-gray-200/70 hover:from-gray-200/70 hover:to-gray-300/70'
                    }`}
                  >
                    <div className="flex justify-between items-start flex-wrap gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2 flex-wrap">
                          <p className="font-black text-xl md:text-2xl">{trade.symbol}</p>
                          <span className={`px-4 py-1 rounded-full text-sm font-bold ${
                            trade.action==='BUY'
                              ?'bg-green-500/30 text-green-300 border border-green-500/50'
                              :trade.action==='SELL'
                                ?'bg-red-500/30 text-red-300 border border-red-500/50'
                                :'bg-yellow-500/30 text-yellow-300 border border-yellow-500/50'
                          }`}>
                            {trade.action==='BUY'&&'📈 '}{trade.action==='SELL'&&'📉 '}{trade.action}
                          </span>
                        </div>
                        <p className={`text-sm mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {trade.date} at {trade.time}
                        </p>
                        <div className="flex items-center gap-4 flex-wrap">
                          <p className="text-lg font-bold text-indigo-400">${trade.price?.toFixed(2)}</p>
                          {trade.shares && (
                            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                              {trade.shares.toFixed(4)} shares
                            </p>
                          )}
                          {trade.profitLoss !== undefined && (
                            <p className={`text-sm font-bold ${trade.profitLoss>=0?'text-green-400':'text-red-400'}`}>
                              P/L: {trade.profitLoss>=0?'+':''}${trade.profitLoss.toFixed(2)}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                    {trade.reason && (
                      <div className={`mt-4 pt-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-300'}`}>
                        <p className={`text-xs font-semibold mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          📝 REASONING:
                        </p>
                        <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                          {trade.reason}
                        </p>
                      </div>
                    )}
                  </div>
                )) : (
                  <div className={`text-center py-20 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                    <p className="text-6xl mb-4">📜</p>
                    <p className="text-xl">No trading history</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab==='about' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-gradient-to-br from-indigo-900/50 to-purple-900/50 backdrop-blur-xl border border-indigo-500/30 rounded-3xl p-8 md:p-10 shadow-2xl">
              <h1 className="text-4xl md:text-5xl font-black mb-6 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                ⚡ Quantum Trade AI
              </h1>
              <p className="text-lg md:text-xl text-gray-300 mb-8 leading-relaxed">
                Professional AI trading platform with {STOCK_DATABASE.length} stocks.
              </p>
              
              {modelInfo?.type==='LSTM' && (
                <div className="p-6 rounded-2xl border bg-green-500/20 border-green-500/30 mb-6">
                  <h3 className="text-2xl font-bold mb-3 text-white">🧠 LSTM Deep Learning Model</h3>
                  {modelInfo.metadata && (
                    <div className="space-y-2 text-gray-200">
                      <p className="text-lg">
                        <span className="text-gray-400">Accuracy:</span>{' '}
                        <span className="text-green-400 font-bold">
                          {(modelInfo.metadata.r2_score*100).toFixed(2)}% R²
                        </span>
                      </p>
                      <p>
                        <span className="text-gray-400">Mean Error:</span>{' '}
                        <span className="text-yellow-400 font-semibold">
                          ${modelInfo.metadata.real_mae.toFixed(2)}
                        </span>
                      </p>
                      <p>
                        <span className="text-gray-400">Training Samples:</span>{' '}
                        <span className="font-semibold">
                          {modelInfo.metadata.total_samples.toLocaleString()}
                        </span>
                      </p>
                    </div>
                  )}
                </div>
              )}
              
              <div className="bg-black/30 p-6 rounded-2xl border border-indigo-500/20 mb-6">
                <h3 className="text-2xl font-bold mb-4 text-indigo-300">🔒 Enterprise Features</h3>
                <div className="space-y-3 text-gray-300">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">📋</span>
                    <div>
                      <p className="font-bold">Audit Logs</p>
                      <p className="text-sm text-gray-400">
                        Every trade is tracked and logged for compliance and review
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">🛡️</span>
                    <div>
                      <p className="font-bold">Safety Guardrails</p>
                      <p className="text-sm text-gray-400">
                        Built-in risk management prevents bad trades automatically
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">⚡</span>
                    <div>
                      <p className="font-bold">Sub-200ms Latency</p>
                      <p className="text-sm text-gray-400">
                        Lightning-fast AI predictions in real-time
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">🔐</span>
                    <div>
                      <p className="font-bold">Data Security</p>
                      <p className="text-sm text-gray-400">
                        Your portfolio and trades stay private in local storage
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-yellow-600/30 to-orange-600/30 p-6 rounded-2xl border border-yellow-400/30">
                <h3 className="text-2xl font-bold mb-3">⚠️ Disclaimer</h3>
                <p className="text-gray-300 text-sm">Educational platform only. Not financial advice.</p>
              </div>
            </div>
          </div>
        )}
      </main>

      {showModal && signal && (
        <div 
          className="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center z-50 p-4" 
          onClick={() => setShowModal(false)}
        >
          <div 
            className="bg-gradient-to-br from-gray-900 to-gray-800 border border-indigo-500/50 rounded-3xl p-6 md:p-8 max-w-3xl w-full shadow-2xl max-h-[90vh] overflow-y-auto" 
            onClick={e => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl md:text-3xl font-bold">🧠 AI Analysis</h2>
              <button 
                onClick={() => setShowModal(false)} 
                className="text-3xl md:text-4xl hover:text-gray-400 transition"
              >
                ×
              </button>
            </div>
            <div className="space-y-6">
              <div className="bg-gradient-to-r from-indigo-600/30 to-purple-600/30 p-6 rounded-2xl border border-indigo-500/30">
                <p className="text-sm text-gray-300 mb-2 font-semibold">ACTION FOR {symbol}</p>
                <p className="text-5xl md:text-6xl font-black">{signal.decision.action}</p>
              </div>
              <div className="bg-indigo-500/10 border border-indigo-500/30 p-6 rounded-2xl">
                <p className="text-indigo-300 font-semibold mb-3 uppercase text-sm">💡 AI Reasoning</p>
                <p className="leading-relaxed text-base md:text-lg text-gray-200">
                  {signal.decision.reason}
                </p>
              </div>
              <button 
                onClick={() => setShowModal(false)} 
                className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 p-4 rounded-xl font-bold transition text-white"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      <EmailAlertModal 
        isOpen={showEmailModal} 
        onClose={() => setShowEmailModal(false)} 
        symbol={symbol} 
        signal={signal} 
      />
    </div>
  );
}

function App() {
  return (
    <DarkModeProvider>
      <MainApp />
    </DarkModeProvider>
  );
}

export default App;