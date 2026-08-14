import React, { useState, useEffect, useCallback, useMemo, useRef, memo, createContext, useContext } from 'react';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Brush
} from 'recharts';
import axios from 'axios';
// fmtMoney and fmtPct now live in utils/format.js — pulled out so they're
// unit-testable in isolation. See src/utils/format.test.js.
import { fmtMoney, fmtPct } from './utils/format';

// ============================================
// INLINE ICONS — plain SVGs instead of lucide-react so this file has
// zero extra npm install steps. `stroke="currentColor"` picks up
// whatever CSS color is set via className/style on the call site.
// ============================================
const Icon = ({ children, ...props }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    {children}
  </svg>
);
const LayoutDashboard = (props) => (<Icon {...props}><rect x="3" y="3" width="7" height="9" rx="1" /><rect x="14" y="3" width="7" height="5" rx="1" /><rect x="14" y="12" width="7" height="9" rx="1" /><rect x="3" y="16" width="7" height="5" rx="1" /></Icon>);
const Activity = (props) => (<Icon {...props}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></Icon>);
const Star = (props) => (<Icon {...props}><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" /></Icon>);
const Briefcase = (props) => (<Icon {...props}><rect x="2" y="7" width="20" height="14" rx="2" /><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" /></Icon>);
const Clock = (props) => (<Icon {...props}><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></Icon>);
const Info = (props) => (<Icon {...props}><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></Icon>);
const Search = (props) => (<Icon {...props}><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></Icon>);
const Bell = (props) => (<Icon {...props}><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /></Icon>);
const Brain = (props) => (<Icon {...props}><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24A2.5 2.5 0 0 1 9.5 2Z" /><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24A2.5 2.5 0 0 0 14.5 2Z" /></Icon>);
const Flame = (props) => (<Icon {...props}><path d="M8.5 14.5A2.5 2.5 0 0 0 11 17a2.5 2.5 0 0 0 2.5-2.5c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7.5 7.5 0 1 1-15 0c0-1.15.43-2.29 1-3a2.5 2.5 0 0 0 2.5 2.5z" /></Icon>);
const RefreshCw = (props) => (<Icon {...props}><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></Icon>);
const LogOut = (props) => (<Icon {...props}><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" /></Icon>);
const X = (props) => (<Icon {...props}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></Icon>);

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';
const ML_SERVICE_URL = process.env.REACT_APP_ML_SERVICE_URL || 'http://localhost:8000';

// ============================================
// DESIGN TOKENS — matches the reference dashboard exactly:
// near-black navy background, green primary, amber for AI/LSTM
// callouts, JetBrains Mono everywhere, dense small type.
// ============================================
const T = {
  bg: '#07090f',
  card: '#0d1220',
  cardAlt: '#141b2d',
  muted: '#1a2235',
  border: 'rgba(255,255,255,0.07)',
  sidebar: '#0a0d18',
  primary: '#00e599',
  primaryFg: '#07090f',
  amber: '#f59e0b',
  red: '#ef4444',
  blue: '#3b82f6',
  purple: '#a855f7',
  text: '#e2e8f0',
  textMuted: '#5a6481',
};

const GlobalStyle = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');
    .qt-root, .qt-root * { font-family: 'JetBrains Mono', 'Courier New', monospace; }
    .qt-root { background: ${T.bg}; color: ${T.text}; }
    .qt-input:focus { outline: none; border-color: ${T.primary}; }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
    @keyframes qt-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .qt-pulse { animation: qt-pulse 2s ease-in-out infinite; }
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
  `}</style>
);

// ============================================
// AUTHENTICATED AXIOS INSTANCE
// ============================================
const api = axios.create({ baseURL: API_URL });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('authToken');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

let onUnauthorized = () => {};
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) onUnauthorized();
    return Promise.reject(error);
  }
);

// ============================================
// AUTH CONTEXT
// ============================================
const AuthContext = createContext();

const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem('authToken'));
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('authUser');
    return saved ? JSON.parse(saved) : null;
  });

  useEffect(() => {
    onUnauthorized = () => {
      localStorage.removeItem('authToken');
      localStorage.removeItem('authUser');
      setToken(null);
      setUser(null);
    };
  }, []);

  const login = useCallback(async (email, password) => {
    const res = await axios.post(`${API_URL}/api/auth/login`, { email, password });
    localStorage.setItem('authToken', res.data.token);
    localStorage.setItem('authUser', JSON.stringify(res.data.user));
    setToken(res.data.token);
    setUser(res.data.user);
  }, []);

  const register = useCallback(async (email, password) => {
    const res = await axios.post(`${API_URL}/api/auth/register`, { email, password });
    localStorage.setItem('authToken', res.data.token);
    localStorage.setItem('authUser', JSON.stringify(res.data.user));
    setToken(res.data.token);
    setUser(res.data.user);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('authUser');
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ token, user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};

// ============================================
// LOGIN / REGISTER SCREEN
// ============================================
const AuthScreen = () => {
  const { login, register } = useAuth();
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === 'login') await login(email, password);
      else await register(email, password);
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="qt-root min-h-screen flex items-center justify-center p-4">
      <GlobalStyle />
      <div className="w-full max-w-sm rounded-lg p-8" style={{ background: T.card, border: `1px solid ${T.border}` }}>
        <div className="flex items-center gap-2.5 justify-center mb-1">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: T.primary }}>
            <Brain className="w-4 h-4" style={{ color: T.primaryFg }} />
          </div>
          <div className="text-left">
            <p className="text-sm font-bold tracking-tight">QUANTUM TRADE</p>
            <p className="text-[9px] uppercase tracking-widest" style={{ color: T.textMuted }}>AI Trading Platform</p>
          </div>
        </div>
        <p className="text-center mb-8 text-[11px] mt-4" style={{ color: T.textMuted }}>
          {mode === 'login' ? 'Log in to your terminal' : 'Open a new account'}
        </p>

        {error && (
          <div className="rounded p-3 mb-4 text-xs" style={{ background: 'rgba(239,68,68,0.1)', border: `1px solid rgba(239,68,68,0.3)`, color: T.red }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-[10px] uppercase tracking-widest block mb-2" style={{ color: T.textMuted }}>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="qt-input w-full rounded px-3 py-2.5 text-xs"
              style={{ background: T.cardAlt, border: `1px solid ${T.border}`, color: T.text }}
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label className="text-[10px] uppercase tracking-widest block mb-2" style={{ color: T.textMuted }}>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="qt-input w-full rounded px-3 py-2.5 text-xs"
              style={{ background: T.cardAlt, border: `1px solid ${T.border}`, color: T.text }}
              placeholder="At least 8 characters"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded font-bold text-xs transition disabled:opacity-50"
            style={{ background: T.primary, color: T.primaryFg }}
          >
            {loading ? 'Please wait…' : mode === 'login' ? 'Log In' : 'Create Account'}
          </button>
        </form>

        <button
          onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null); }}
          className="w-full text-center text-[11px] mt-6 transition"
          style={{ color: T.primary }}
        >
          {mode === 'login' ? "Don't have an account? Register" : 'Already have an account? Log in'}
        </button>
      </div>
    </div>
  );
};

// ============================================
// STOCK DATABASE
// ============================================
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

const TRENDING_SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'META', 'GOOGL', 'AMZN'];

const TIME_RANGES = [
  { label: '1W', days: 7 }, { label: '1M', days: 30 }, { label: '3M', days: 90 },
  { label: '6M', days: 180 }, { label: '1Y', days: 365 }, { label: '2Y', days: 730 },
  { label: '5Y', days: 1825 }, { label: 'ALL', days: null }
];

// ============================================
// SHARED UI PIECES (mirroring the reference dashboard)
// ============================================
const TickerBadge = memo(({ ticker }) => (
  <div className="w-8 h-8 rounded-md flex items-center justify-center text-[10px] font-bold flex-shrink-0" style={{ background: T.muted, color: T.text }}>
    {ticker.slice(0, 2)}
  </div>
));

const StatCard = memo(({ label, value, sub, positive }) => (
  <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
    <p className="text-[10px] uppercase tracking-widest mb-1" style={{ color: T.textMuted }}>{label}</p>
    <p className="text-xl font-bold leading-tight">{value}</p>
    {sub && (
      <p className="text-xs mt-1" style={{ color: positive === undefined ? T.textMuted : positive ? T.primary : T.red }}>
        {sub}
      </p>
    )}
  </div>
));

// Small two-point trend indicator built from a real latest-vs-previous-close
// comparison (not a fabricated shape) — echoes the reference's sparkline
// without inventing intraday detail we don't have.
const MiniTrend = memo(({ changePct }) => {
  const up = (changePct || 0) >= 0;
  const color = up ? T.primary : T.red;
  return (
    <svg width="48" height="22" viewBox="0 0 48 22">
      <polyline points={up ? '2,18 24,11 46,4' : '2,4 24,11 46,18'} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
});

const SignalBadge = memo(({ action }) => {
  const styles = {
    BUY: { bg: 'rgba(0,229,153,0.1)', color: T.primary },
    SELL: { bg: 'rgba(239,68,68,0.1)', color: T.red },
    HOLD: { bg: 'rgba(245,158,11,0.1)', color: T.amber },
  };
  const s = styles[action] || styles.HOLD;
  return (
    <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: s.bg, color: s.color }}>
      <Brain className="w-2.5 h-2.5" />
      {action}
    </span>
  );
});

const ModelStatusBadge = memo(({ modelInfo }) => {
  if (!modelInfo) return null;
  const isLSTM = modelInfo.lstm_loaded === true;
  return (
    <div className="px-2.5 py-1 rounded flex items-center gap-1.5" style={{ background: isLSTM ? 'rgba(0,229,153,0.1)' : 'rgba(245,158,11,0.1)', border: `1px solid ${isLSTM ? 'rgba(0,229,153,0.25)' : 'rgba(245,158,11,0.25)'}` }}>
      <div className="w-1.5 h-1.5 rounded-full" style={{ background: isLSTM ? T.primary : T.amber }} />
      <span className="text-[10px] font-semibold">{isLSTM ? 'LSTM ACTIVE' : 'MODEL UNAVAILABLE'}</span>
    </div>
  );
});

const CustomTooltip = memo(({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="rounded px-3 py-2 text-xs shadow-lg" style={{ background: T.card, border: `1px solid ${T.border}` }}>
        <p className="mb-1" style={{ color: T.textMuted }}>{data.fullDate || label}</p>
        {data.open !== undefined ? (
          <div className="space-y-0.5">
            <p>Open: {fmtMoney(data.open)}</p>
            <p style={{ color: T.primary }}>High: {fmtMoney(data.high)}</p>
            <p style={{ color: T.red }}>Low: {fmtMoney(data.low)}</p>
            <p>Close: {fmtMoney(data.price)}</p>
          </div>
        ) : (
          <p className="font-bold">{data.isPrediction && '◆ '}{fmtMoney(payload[0].value)}</p>
        )}
        {data.volume ? <p className="text-[10px] mt-1 pt-1 border-t" style={{ color: T.textMuted, borderColor: T.border }}>Vol: {(data.volume / 1000000).toFixed(2)}M</p> : null}
      </div>
    );
  }
  return null;
});

// ============================================
// CANDLESTICK CHART — self-measuring, own axes.
// Doesn't rely on recharts' function-as-children ResponsiveContainer
// pattern (fragile — renders nothing on some builds/versions).
// ============================================
const Candlestick = memo(({ data, height = 260 }) => {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width);
    });
    ro.observe(el);
    setWidth(el.getBoundingClientRect().width);
    return () => ro.disconnect();
  }, []);

  const validData = useMemo(
    () => (data || []).filter(d => d.open != null && d.high != null && d.low != null && d.price != null),
    [data]
  );

  const axisWidth = 52;
  const bottomAxisHeight = 28;
  const topPad = 14;
  const plotWidth = Math.max(0, width - axisWidth);
  const plotHeight = height - bottomAxisHeight;

  const maxPrice = validData.length ? Math.max(...validData.map(d => d.high)) : 0;
  const minPrice = validData.length ? Math.min(...validData.map(d => d.low)) : 0;
  const priceRange = (maxPrice - minPrice) || 1;
  const getY = (price) => topPad + ((maxPrice - price) / priceRange) * (plotHeight - topPad * 2);

  const barSpacing = validData.length ? plotWidth / validData.length : 0;
  const barWidth = Math.max(2, Math.min(10, barSpacing * 0.6));

  const tickCount = 5;
  const priceTicks = Array.from({ length: tickCount }, (_, i) => minPrice + (priceRange * i) / (tickCount - 1));
  const labelEvery = Math.max(1, Math.ceil(validData.length / 8));

  return (
    <div ref={containerRef} style={{ width: '100%', height }}>
      {width > 0 && validData.length > 0 && (
        <svg width={width} height={height}>
          {priceTicks.map((tv, i) => (
            <g key={i}>
              <line x1={axisWidth} y1={getY(tv)} x2={width} y2={getY(tv)} stroke={T.border} strokeDasharray="3 3" />
              <text x={axisWidth - 8} y={getY(tv) + 4} textAnchor="end" fontSize={9} fill={T.textMuted}>
                {`$${tv.toFixed(0)}`}
              </text>
            </g>
          ))}
          {validData.map((item, index) => {
            const x = axisWidth + index * barSpacing + barSpacing / 2;
            const isGreen = item.price >= item.open;
            const color = isGreen ? T.primary : T.red;
            return (
              <g key={index}>
                <line x1={x} y1={getY(item.high)} x2={x} y2={getY(item.low)} stroke={color} strokeWidth={1.5} />
                <rect
                  x={x - barWidth / 2}
                  y={Math.min(getY(item.open), getY(item.price))}
                  width={barWidth}
                  height={Math.max(Math.abs(getY(item.price) - getY(item.open)), 1.5)}
                  fill={color}
                  stroke={color}
                  strokeWidth={1}
                />
                {index % labelEvery === 0 && (
                  <text x={x} y={plotHeight + 20} textAnchor="middle" fontSize={9} fill={T.textMuted}>
                    {item.date}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
});

const EmailAlertModal = memo(({ isOpen, onClose, symbol, signal }) => {
  const [email, setEmail] = useState(() => localStorage.getItem('userEmail') || '');
  const [sending, setSending] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSendAlert = async () => {
    if (!email || !email.includes('@')) { alert('Please enter a valid email address'); return; }
    setSending(true);
    try {
      localStorage.setItem('userEmail', email);
      await api.post('/api/send-alert', {
        email, symbol, action: signal.decision.action, price: signal.current_price, reason: signal.decision.reason
      });
      setSuccess(true);
      setTimeout(() => { setSuccess(false); onClose(); }, 2000);
    } catch (error) {
      alert('Failed to send email. Backend email service may not be configured.');
    } finally {
      setSending(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="rounded-lg p-6 max-w-sm w-full" onClick={e => e.stopPropagation()} style={{ background: T.card, border: `1px solid ${T.primary}` }}>
        <h3 className="text-sm font-bold mb-4">Email this signal</h3>
        <div className="mb-4">
          <label className="text-[10px] uppercase tracking-widest block mb-2" style={{ color: T.textMuted }}>Your email</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="your@email.com"
            className="qt-input w-full rounded px-3 py-2 text-xs" style={{ background: T.cardAlt, border: `1px solid ${T.border}`, color: T.text }} />
        </div>
        <div className="rounded p-3 mb-5 text-xs" style={{ background: T.cardAlt, border: `1px solid ${T.border}` }}>
          <p className="mb-1" style={{ color: T.textMuted }}>
            Signal: <strong style={{ color: T.text }}>{signal?.decision.action}</strong> {symbol}
          </p>
          <p style={{ color: T.textMuted }}>
            Price: <strong style={{ color: T.text }}>{fmtMoney(signal?.current_price)}</strong>
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={handleSendAlert} disabled={sending || !email}
            className="flex-1 py-2.5 rounded font-bold text-xs transition disabled:opacity-50"
            style={{ background: T.primary, color: T.primaryFg }}>
            {sending ? 'Sending…' : success ? 'Sent' : 'Send Alert'}
          </button>
          <button onClick={onClose} className="px-5 py-2.5 rounded font-bold text-xs transition"
            style={{ background: T.cardAlt, border: `1px solid ${T.border}` }}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
});

// ============================================
// SIDEBAR
// ============================================
const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'analysis', label: 'Analysis', icon: Activity },
  { id: 'watchlist', label: 'Watchlist', icon: Star },
  { id: 'portfolio', label: 'Portfolio', icon: Briefcase },
  { id: 'history', label: 'History', icon: Clock },
  { id: 'about', label: 'About', icon: Info },
];

const Sidebar = memo(({ tab, setTab, mobileOpen, closeMobile }) => {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 30000);
    return () => clearInterval(t);
  }, []);

  return (
    <>
      {/* Backdrop — mobile only, closes the drawer on tap outside it */}
      {mobileOpen && (
        <div className="fixed inset-0 bg-black/60 z-40 lg:hidden" onClick={closeMobile} />
      )}
      <aside
        className={`w-64 lg:w-52 flex-shrink-0 flex flex-col fixed lg:static inset-y-0 left-0 z-50 transition-transform duration-200 ${mobileOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0`}
        style={{ background: T.sidebar, borderRight: `1px solid ${T.border}` }}
      >
        <div className="px-4 py-4 flex items-center justify-between" style={{ borderBottom: `1px solid ${T.border}` }}>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: T.primary }}>
              <Brain className="w-4 h-4" style={{ color: T.primaryFg }} />
            </div>
            <div>
              <p className="text-sm font-bold tracking-tight">QUANTUM TRADE</p>
              <p className="text-[9px] tracking-widest uppercase" style={{ color: T.textMuted }}>AI Trading Platform</p>
            </div>
          </div>
          <button onClick={closeMobile} className="lg:hidden p-1" style={{ color: T.textMuted }}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <nav className="flex-1 px-2 py-3 space-y-0.5">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => { setTab(id); closeMobile(); }}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-xs transition-all duration-150"
              style={tab === id
                ? { background: 'rgba(0,229,153,0.1)', color: T.primary, border: `1px solid rgba(0,229,153,0.2)` }
                : { color: T.textMuted, border: '1px solid transparent' }}
            >
              <Icon className="w-3.5 h-3.5 flex-shrink-0" />
              <span className="font-medium">{label}</span>
              {tab === id && <span className="ml-auto w-1.5 h-1.5 rounded-full" style={{ background: T.primary }} />}
            </button>
          ))}
        </nav>

        <div className="px-4 py-3 space-y-1" style={{ borderTop: `1px solid ${T.border}` }}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full qt-pulse" style={{ background: T.primary }} />
            <span className="text-[10px]" style={{ color: T.textMuted }}>Session active</span>
          </div>
          <p className="text-[9px]" style={{ color: T.textMuted }}>
            {now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} · {now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
      </aside>
    </>
  );
});

// ============================================
// TOP BAR
// ============================================
const TopBar = memo(({
  user, logout, error, searchQuery, setSearchQuery, filteredStocks,
  onSelectSymbol, watchlist, stockQuotes, onOpenMobileMenu,
}) => {
  const items = watchlist.length > 0 ? watchlist.slice(0, 7) : ['AAPL', 'MSFT', 'GOOGL'];
  const initials = (user?.email || '??').slice(0, 2).toUpperCase();

  return (
    <header className="h-12 flex items-center px-3 sm:px-5 gap-3 sm:gap-4 flex-shrink-0 relative" style={{ background: 'rgba(13,18,32,0.6)', backdropFilter: 'blur(6px)', borderBottom: `1px solid ${T.border}` }}>
      <button onClick={onOpenMobileMenu} className="lg:hidden p-1 flex-shrink-0" style={{ color: T.textMuted }} aria-label="Open menu">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="w-5 h-5">
          <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <div className="relative flex-shrink-0">
        <div className="flex items-center gap-2 rounded px-2.5 py-1.5 w-28 sm:w-44" style={{ background: T.cardAlt, border: `1px solid ${T.border}` }}>
          <Search className="w-3 h-3 flex-shrink-0" style={{ color: T.textMuted }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search…"
            className="bg-transparent text-xs outline-none w-full"
            style={{ color: T.text }}
          />
        </div>
        {searchQuery && filteredStocks.length > 0 && (
          <div className="absolute top-full left-0 mt-1 w-72 max-w-[85vw] max-h-72 overflow-y-auto rounded-lg z-30" style={{ background: T.card, border: `1px solid ${T.border}` }}>
            {filteredStocks.slice(0, 10).map(stock => (
              <button key={stock.symbol} onClick={() => onSelectSymbol(stock.symbol)}
                className="w-full text-left px-3 py-2 flex items-center gap-2 hover:bg-white/5 transition-colors">
                <TickerBadge ticker={stock.symbol} />
                <div className="min-w-0">
                  <p className="text-xs font-bold">{stock.symbol}</p>
                  <p className="text-[10px] truncate" style={{ color: T.textMuted }}>{stock.name}</p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1 flex items-center gap-5 overflow-x-auto no-scrollbar">
        {items.map(sym => {
          const q = stockQuotes[sym];
          return (
            <div key={sym} className="flex items-center gap-1.5 flex-shrink-0 text-[11px]">
              <span className="font-medium" style={{ color: T.textMuted }}>{sym}</span>
              <span>{q ? fmtMoney(q.price) : '—'}</span>
              {q && <span style={{ color: q.changePct >= 0 ? T.primary : T.red }}>{fmtPct(q.changePct)}</span>}
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-2.5">
        <button className="relative p-1.5 transition-colors" style={{ color: T.textMuted }} title={error || 'No alerts'}>
          <Bell className="w-3.5 h-3.5" />
          {error && <span className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full" style={{ background: T.red }} />}
        </button>
        <div className="w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold" style={{ background: 'rgba(0,229,153,0.15)', border: `1px solid rgba(0,229,153,0.3)`, color: T.primary }} title={user?.email}>
          {initials}
        </div>
        <button onClick={logout} className="p-1.5 transition-colors" style={{ color: T.textMuted }} title="Log out">
          <LogOut className="w-3.5 h-3.5" />
        </button>
      </div>
    </header>
  );
});

function MainApp() {
  const { user, logout } = useAuth();

  // Namespaced by user id — without this, watchlist/lastSymbol were
  // shared browser-wide via bare localStorage keys, so logging into a
  // second account on the same machine would inherit the first
  // account's watchlist and last-viewed stock. Portfolio/holdings were
  // never affected (those come from the DB, filtered by user_id) — only
  // these two client-only preferences were leaking.
  const watchlistKey = `watchlist_${user.id}`;
  const lastSymbolKey = `lastSymbol_${user.id}`;

  const [symbol, setSymbol] = useState(() => localStorage.getItem(lastSymbolKey) || 'AAPL');
  const [searchQuery, setSearchQuery] = useState('');
  const [prices, setPrices] = useState([]);
  const [pricesLoading, setPricesLoading] = useState(false);
  const [signal, setSignal] = useState(null);
  const [prediction, setPrediction] = useState(null);
  // { count, avgAbsPctError, directionAccuracy, history } for the current
  // symbol — how past predictions compared to what actually happened.
  const [accuracy, setAccuracy] = useState(null);
  const [accuracyLoading, setAccuracyLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);

  const [portfolio, setPortfolio] = useState(null);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [tradeSubmitting, setTradeSubmitting] = useState(false);

  const [watchlist, setWatchlist] = useState(() => {
    const saved = localStorage.getItem(watchlistKey);
    const parsed = saved ? JSON.parse(saved) : ['AAPL', 'GOOGL', 'MSFT'];
    // Defensive: strip any symbol no longer in STOCK_DATABASE (e.g. an old
    // BTC/crypto entry saved before crypto was removed) so stale
    // localStorage can never resurrect a dead symbol.
    return parsed.filter(sym => STOCK_DATABASE.some(s => s.symbol === sym));
  });

  const [showEmailModal, setShowEmailModal] = useState(false);
  const [chartType, setChartType] = useState('area');
  // [startIndex, endIndex] into filteredChartData, or null = full range.
  // Driven by the Brush control below the chart.
  const [zoomRange, setZoomRange] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [error, setError] = useState(null);
  // symbol -> { price, changePct } — changePct derived from real latest-vs-previous close
  const [stockQuotes, setStockQuotes] = useState({});
  const [manualTradeAmount, setManualTradeAmount] = useState('');
  const [selectedTimeRange, setSelectedTimeRange] = useState('3M');
  const [customStartDate, setCustomStartDate] = useState('');
  const [customEndDate, setCustomEndDate] = useState('');
  const [useCustomRange, setUseCustomRange] = useState(false);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const res = await axios.get(`${ML_SERVICE_URL}/health`);
        setModelInfo(res.data);
      } catch (err) {
        console.warn('Could not fetch ML model info:', err.message);
      }
    };
    fetchModelInfo();
  }, []);

  useEffect(() => { localStorage.setItem(lastSymbolKey, symbol); }, [symbol, lastSymbolKey]);
  useEffect(() => { localStorage.setItem(watchlistKey, JSON.stringify(watchlist)); }, [watchlist, watchlistKey]);


  const fetchPortfolio = useCallback(async () => {
    try {
      const res = await api.get('/api/portfolio');
      if (res.data?.success) setPortfolio(res.data);
    } catch (e) {
      console.error('Portfolio fetch error:', e);
    }
  }, []);

  const fetchTradeHistory = useCallback(async () => {
    try {
      const res = await api.get('/api/trades');
      if (res.data?.success) setTradeHistory(res.data.data);
    } catch (e) {
      console.error('Trade history fetch error:', e);
    }
  }, []);

  useEffect(() => { fetchPortfolio(); fetchTradeHistory(); }, [fetchPortfolio, fetchTradeHistory]);

  const filteredStocks = useMemo(() =>
    searchQuery ? STOCK_DATABASE.filter(s =>
      s.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.category.toLowerCase().includes(searchQuery.toLowerCase())
    ) : [], [searchQuery]
  );

  const loadPrices = useCallback(async () => {
    try {
      setPricesLoading(true);
      setError(null);
      const res = await api.get(`/api/prices/${symbol}?limit=2000`);
      if (res.data?.success) {
        const priceData = res.data.data.reverse();
        setPrices(priceData);
        if (priceData.length > 0) {
          const last = parseFloat(priceData[priceData.length - 1].close);
          const prev = priceData.length > 1 ? parseFloat(priceData[priceData.length - 2].close) : last;
          const changePct = prev ? ((last - prev) / prev) * 100 : 0;
          setStockQuotes(prevQ => ({ ...prevQ, [symbol]: { price: last, changePct } }));
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

  const refreshQuotes = useCallback(async () => {
    const symbolsToRefresh = [...new Set([
      ...watchlist,
      ...TRENDING_SYMBOLS,
      ...(portfolio ? Object.keys(portfolio.portfolio || {}) : []),
    ])];
    if (symbolsToRefresh.length === 0) return;
    const results = await Promise.all(
      symbolsToRefresh.map(async (sym) => {
        try {
          const res = await api.get(`/api/prices/${sym}?limit=2`);
          if (res.data?.success && res.data.data.length > 0) {
            const latest = parseFloat(res.data.data[0].close);
            const prevClose = res.data.data.length > 1 ? parseFloat(res.data.data[1].close) : latest;
            const changePct = prevClose ? ((latest - prevClose) / prevClose) * 100 : 0;
            return [sym, { price: latest, changePct }];
          }
        } catch (e) { /* ignore, keep prior value */ }
        return null;
      })
    );
    setStockQuotes(prev => {
      const updated = { ...prev };
      results.forEach(r => { if (r) updated[r[0]] = r[1]; });
      return updated;
    });
  }, [watchlist, portfolio]);

  useEffect(() => { refreshQuotes(); }, [refreshQuotes]);

  const currentPrice = useMemo(() => prices.length > 0 ? parseFloat(prices[prices.length - 1].close) : 0, [prices]);
  const priceChange = useMemo(() => prices.length > 1
    ? ((currentPrice - parseFloat(prices[prices.length - 2].close)) / parseFloat(prices[prices.length - 2].close) * 100)
    : 0, [prices, currentPrice]);
  const latestVolume = useMemo(() => prices.length > 0 ? parseInt(prices[prices.length - 1].volume) : 0, [prices]);

  // Chart data — dates formatted as readable "MMM D" labels (with a
  // 2-digit year appended when the visible range spans more than one
  // calendar year) instead of a bare day-of-month number that resets
  // every month and looks scrambled across multi-month ranges.
  const filteredChartData = useMemo(() => {
    if (!prices.length) return [];
    let filteredPrices = [...prices];
    if (useCustomRange && customStartDate && customEndDate) {
      const start = new Date(customStartDate);
      const end = new Date(customEndDate);
      filteredPrices = prices.filter(p => { const d = new Date(p.timestamp); return d >= start && d <= end; });
    } else {
      const range = TIME_RANGES.find(r => r.label === selectedTimeRange);
      if (range && range.days && prices.length > 0) {
        // Count back from the NEWEST date actually in the data, not the
        // real-world clock — otherwise short ranges (1M, 3M, 6M) can
        // silently exclude everything whenever the DB's latest row is
        // older than "today - range.days" in real time (e.g. the daily
        // updater hasn't run yet). 1Y/2Y/5Y/ALL never hit this because
        // they're wide enough to always include the latest data anyway.
        const latestDataDate = new Date(Math.max(...prices.map(p => new Date(p.timestamp).getTime())));
        const cutoffDate = new Date(latestDataDate);
        cutoffDate.setDate(cutoffDate.getDate() - range.days);
        filteredPrices = prices.filter(p => new Date(p.timestamp) >= cutoffDate);
      }
    }

    const spansMultipleYears = new Set(filteredPrices.map(p => new Date(p.timestamp).getFullYear())).size > 1;
    const dateFormat = spansMultipleYears
      ? { month: 'short', day: 'numeric', year: '2-digit' }
      : { month: 'short', day: 'numeric' };

    const data = filteredPrices.map((p) => {
      const date = new Date(p.timestamp);
      return {
        date: date.toLocaleDateString('en-US', dateFormat),
        fullDate: date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
        price: parseFloat(p.close), volume: parseInt(p.volume),
        high: parseFloat(p.high), low: parseFloat(p.low), open: parseFloat(p.open),
      };
    });
    if (prediction && data.length > 0) {
      const lastDate = new Date(filteredPrices[filteredPrices.length - 1].timestamp);
      lastDate.setDate(lastDate.getDate() + 1);
      data.push({ date: 'F', fullDate: 'AI Prediction - ' + lastDate.toLocaleDateString('en-US'), price: prediction.predicted_price, isPrediction: true });
    }
    return data;
  }, [prices, prediction, selectedTimeRange, customStartDate, customEndDate, useCustomRange]);

  // The slice actually rendered by the main chart, volume chart, and
  // candlestick once the person drags the Brush handles below the
  // chart. null zoomRange (default / just reset) means "show everything
  // in filteredChartData" — zoomedChartData is what every chart type
  // reads from so zoom behaves identically for area/line/bar/candlestick.
  const zoomedChartData = useMemo(() => {
    if (!zoomRange) return filteredChartData;
    return filteredChartData.slice(zoomRange[0], zoomRange[1] + 1);
  }, [filteredChartData, zoomRange]);

  // Reset zoom whenever the time range, symbol, or custom-range toggle
  // changes — otherwise an old zoom window can point at indices that no
  // longer exist in the new filteredChartData, or silently slice the
  // wrong stock's data right after switching symbols.
  useEffect(() => {
    setZoomRange(null);
  }, [selectedTimeRange, symbol, useCustomRange]);

  // Skip labels so long ranges don't cram hundreds of ticks together.
  // Based on the zoomed slice, since that's what's actually on screen.
  const xAxisTickInterval = useMemo(
    () => Math.max(0, Math.ceil(zoomedChartData.length / 9) - 1),
    [zoomedChartData.length]
  );

  // Year range shown under the chart — deliberately based on
  // filteredChartData (what the selected time range/custom dates
  // actually cover), not the full `prices` history. Using the full
  // history here made a 3M or 1M selection misleadingly show "2021 –
  // 2026" underneath a chart that only displays 3 months.
  const filteredYearRange = useMemo(() => {
    const realPoints = filteredChartData.filter(d => !d.isPrediction && d.fullDate);
    if (!realPoints.length) return null;
    const years = realPoints.map(d => new Date(d.fullDate).getFullYear());
    const min = Math.min(...years);
    const max = Math.max(...years);
    return min === max ? `${min}` : `${min} – ${max}`;
  }, [filteredChartData]);

  const generateSimpleReasoning = useCallback((stockSymbol, action) => {
    const stock = STOCK_DATABASE.find(s => s.symbol === stockSymbol);
    const stockName = stock?.name || stockSymbol;
    const explanations = {
      'BUY': `Momentum for ${stockName} is trending upward — the model sees strengthening signals over the recent window.`,
      'SELL': `${stockName} is showing downward pressure — recent signals point to weakening momentum.`,
      'HOLD': `${stockName} is in a neutral zone — no strong directional signal right now.`,
    };
    return explanations[action] || explanations.HOLD;
  }, []);

  const executeTrade = useCallback(async (action) => {
    const qty = parseFloat(manualTradeAmount);
    if (!manualTradeAmount || isNaN(qty) || qty <= 0) { setError('Enter a valid share quantity'); return; }

    setTradeSubmitting(true);
    setError(null);
    try {
      await api.post('/api/trade/execute', { symbol, action, quantity: qty });
      setManualTradeAmount('');
      await Promise.all([fetchPortfolio(), fetchTradeHistory()]);
    } catch (e) {
      setError(e.response?.data?.error || 'Trade failed');
    } finally {
      setTradeSubmitting(false);
    }
  }, [manualTradeAmount, symbol, fetchPortfolio, fetchTradeHistory]);

  const exportTradeHistory = useCallback(() => {
    if (tradeHistory.length === 0) return;
    const csv = ['Date,Symbol,Action,Price,Quantity,P/L', ...tradeHistory.map(t =>
      `${new Date(t.timestamp).toLocaleString()},${t.symbol},${t.action},${parseFloat(t.price).toFixed(2)},${parseFloat(t.quantity).toFixed(4)},${t.profit_loss ? parseFloat(t.profit_loss).toFixed(2) : ''}`
    )].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trades-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [tradeHistory]);

  const fetchAccuracy = useCallback(async (sym) => {
    setAccuracyLoading(true);
    try {
      const res = await api.get(`/api/predictions/accuracy/${sym}`);
      if (res.data?.success) setAccuracy(res.data);
      else setAccuracy(null);
    } catch (e) {
      // Not fatal — accuracy is a nice-to-have panel, not core functionality.
      setAccuracy(null);
    } finally {
      setAccuracyLoading(false);
    }
  }, []);

  useEffect(() => { fetchAccuracy(symbol); }, [symbol, fetchAccuracy]);

  const handleAnalyze = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [predRes, tradeRes] = await Promise.all([
        api.post('/api/predict', { symbol }).catch((e) => ({ data: { success: false }, _err: e })),
        api.post('/api/trade', { symbol, balance: portfolio?.balance || 10000, shares: portfolio?.portfolio?.[symbol]?.shares || 0 }).catch((e) => ({ data: { success: false }, _err: e }))
      ]);
      if (predRes.data?.success) {
        setPrediction(predRes.data.prediction);
        // The backend logs this prediction server-side (see /api/predict) so
        // it can be scored later once the actual close lands via the daily
        // sync. Re-fetching now picks up any older predictions that just
        // became scoreable, even though today's won't have an actual yet.
        fetchAccuracy(symbol);
      } else {
        setPrediction(null);
        const detail = predRes._err?.response?.data?.error;
        if (detail) setError(`Prediction unavailable: ${detail}`);
      }
      if (tradeRes.data?.success) {
        setSignal({ ...tradeRes.data, decision: { ...tradeRes.data.decision, reason: generateSimpleReasoning(symbol, tradeRes.data.decision.action) } });
      } else {
        setSignal(null);
      }
    } catch (error) {
      setError('AI analysis failed');
    } finally {
      setLoading(false);
    }
  }, [symbol, portfolio, generateSimpleReasoning, fetchAccuracy]);

  const toggleWatchlist = useCallback((sym) => {
    setWatchlist(prev => prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]);
  }, []);

  const jumpToSymbol = useCallback((sym) => {
    setSymbol(sym);
    setSearchQuery('');
    setSignal(null);
    setPrediction(null);
    setActiveTab('analysis');
  }, []);

  // Recharts' Brush fires onChange on every pixel of drag. Setting state
  // synchronously on each of those events can round-trip back into the
  // Brush's own internal setState fast enough to blow React's nested-update
  // limit ("Maximum update depth exceeded"). Coalescing to one state update
  // per animation frame, and skipping the update entirely when the indices
  // haven't actually changed, breaks that loop.
  const zoomFrameRef = useRef(null);
  const handleBrushChange = useCallback((range) => {
    if (zoomFrameRef.current) cancelAnimationFrame(zoomFrameRef.current);
    zoomFrameRef.current = requestAnimationFrame(() => {
      setZoomRange(prev => {
        if (prev && prev[0] === range.startIndex && prev[1] === range.endIndex) return prev;
        return [range.startIndex, range.endIndex];
      });
    });
  }, []);
  useEffect(() => () => { if (zoomFrameRef.current) cancelAnimationFrame(zoomFrameRef.current); }, []);

  const renderMainChart = useCallback((height) => {
    if (chartType === 'candlestick') {
      return <Candlestick data={zoomedChartData} height={height} />;
    }
    const commonProps = { data: zoomedChartData, margin: { top: 10, right: 20, left: 0, bottom: 0 } };
    const xAxisProps = {
      dataKey: 'date', tick: { fill: T.textMuted, fontSize: 9 }, axisLine: false, tickLine: false,
      interval: xAxisTickInterval, minTickGap: 12,
    };
    const yAxisProps = { tick: { fill: T.textMuted, fontSize: 9 }, axisLine: false, tickLine: false, width: 46, tickFormatter: (v) => `$${v.toFixed(0)}` };
    const shared = (<><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis {...xAxisProps} /><YAxis {...yAxisProps} /><Tooltip content={<CustomTooltip />} /></>);
    if (chartType === 'area') {
      return (<ResponsiveContainer width="100%" height={height}><AreaChart {...commonProps}><defs><linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={T.primary} stopOpacity={0.15} /><stop offset="95%" stopColor={T.primary} stopOpacity={0} /></linearGradient></defs>{shared}<Area type="monotone" dataKey="price" stroke={T.primary} strokeWidth={2} fillOpacity={1} fill="url(#colorPrice)" dot={false} /></AreaChart></ResponsiveContainer>);
    } else if (chartType === 'line') {
      return (<ResponsiveContainer width="100%" height={height}><LineChart {...commonProps}>{shared}<Line type="monotone" dataKey="price" stroke={T.primary} strokeWidth={2} dot={false} activeDot={{ r: 4 }} /></LineChart></ResponsiveContainer>);
    } else {
      return (<ResponsiveContainer width="100%" height={height}><BarChart {...commonProps}>{shared}<Bar dataKey="price" fill={T.primary} radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer>);
    }
  }, [chartType, zoomedChartData, xAxisTickInterval]);

  const balance = portfolio?.balance ?? 0;
  const totalValue = portfolio?.totalValue ?? 0;
  const totalProfitLoss = portfolio?.totalProfitLoss ?? 0;
  const holdingsMap = useMemo(() => portfolio?.portfolio || {}, [portfolio]);
  const currentSymbolShares = holdingsMap[symbol]?.shares || 0;
  const holdingsCount = Object.keys(holdingsMap).length;

  const pieData = useMemo(() => {
    const entries = Object.entries(holdingsMap).map(([sym, h]) => ({ name: sym, value: h.currentValue || 0 }));
    entries.push({ name: 'Cash', value: balance });
    return entries;
  }, [holdingsMap, balance]);
  const PIE_COLORS = [T.primary, T.amber, T.blue, T.purple, T.textMuted];

  return (
    <div className="qt-root flex h-screen overflow-hidden">
      <GlobalStyle />
      <Sidebar tab={activeTab} setTab={setActiveTab} mobileOpen={mobileMenuOpen} closeMobile={() => setMobileMenuOpen(false)} />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar
          user={user} logout={logout} error={error}
          searchQuery={searchQuery} setSearchQuery={setSearchQuery} filteredStocks={filteredStocks}
          onSelectSymbol={jumpToSymbol} watchlist={watchlist} stockQuotes={stockQuotes}
          onOpenMobileMenu={() => setMobileMenuOpen(true)}
        />

        {error && (
          <div className="px-5 pt-3 flex-shrink-0">
            <div className="rounded-lg px-4 py-2.5 flex justify-between items-center text-xs" style={{ background: 'rgba(239,68,68,0.08)', border: `1px solid rgba(239,68,68,0.25)` }}>
              <p style={{ color: T.red }}>{error}</p>
              <button onClick={() => setError(null)} style={{ color: T.red }}><X className="w-3.5 h-3.5" /></button>
            </div>
          </div>
        )}

        <main className="flex-1 overflow-y-auto overflow-x-hidden">

          {/* ─── DASHBOARD ─── */}
          {activeTab === 'dashboard' && (
            <div className="p-5 space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-lg font-bold">Dashboard</h1>
                  <p className="text-[10px] mt-0.5" style={{ color: T.textMuted }}>{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
                </div>
                <button onClick={() => { loadPrices(); fetchPortfolio(); fetchTradeHistory(); refreshQuotes(); }}
                  className="flex items-center gap-1.5 text-[11px] transition-colors" style={{ color: T.textMuted }}>
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <StatCard label="Portfolio Value" value={fmtMoney(totalValue)}
                  sub={`${totalProfitLoss >= 0 ? '+' : ''}${fmtMoney(totalProfitLoss)} all-time`} positive={totalProfitLoss >= 0} />
                <StatCard label="Cash Balance" value={fmtMoney(balance)} sub={`${holdingsCount} open position${holdingsCount === 1 ? '' : 's'}`} />
                <StatCard label="Total P/L" value={`${totalProfitLoss >= 0 ? '+' : ''}${fmtMoney(totalProfitLoss)}`}
                  sub={portfolio?.initialBalance ? fmtPct((totalProfitLoss / portfolio.initialBalance) * 100) : '—'} positive={totalProfitLoss >= 0} />
                <StatCard label="ML Model" value={modelInfo?.lstm_loaded ? 'Active' : 'Unavailable'} sub={modelInfo?.lstm_loaded ? 'LSTM ready' : 'Check ML service'} positive={modelInfo?.lstm_loaded} />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="lg:col-span-2 rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <p className="text-[10px] uppercase tracking-widest" style={{ color: T.textMuted }}>{symbol} — {selectedTimeRange}</p>
                      <p className="text-base font-bold mt-0.5">{fmtMoney(currentPrice)}</p>
                    </div>
                    <span className="text-[11px] px-2 py-1 rounded" style={{ color: priceChange >= 0 ? T.primary : T.red, background: priceChange >= 0 ? 'rgba(0,229,153,0.1)' : 'rgba(239,68,68,0.1)', border: `1px solid ${priceChange >= 0 ? 'rgba(0,229,153,0.2)' : 'rgba(239,68,68,0.2)'}` }}>
                      {fmtPct(priceChange)} today
                    </span>
                  </div>
                  {pricesLoading ? (
                    <div className="h-[150px] flex items-center justify-center text-xs" style={{ color: T.textMuted }}>Loading…</div>
                  ) : filteredChartData.length === 0 ? (
                    <div className="h-[150px] flex items-center justify-center text-xs" style={{ color: T.textMuted }}>No data for {symbol}</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={150}>
                      <AreaChart data={filteredChartData}>
                        <defs>
                          <linearGradient id="dashGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={T.primary} stopOpacity={0.15} />
                            <stop offset="95%" stopColor={T.primary} stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                        <XAxis dataKey="date" tick={{ fontSize: 9, fill: T.textMuted }} axisLine={false} tickLine={false} interval={xAxisTickInterval} />
                        <YAxis tick={{ fontSize: 9, fill: T.textMuted }} axisLine={false} tickLine={false} domain={['auto', 'auto']} tickFormatter={v => `$${v.toFixed(0)}`} width={40} />
                        <Tooltip content={<CustomTooltip />} />
                        <Area type="monotone" dataKey="price" stroke={T.primary} fill="url(#dashGrad)" strokeWidth={2} dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>

                <div className="rounded-lg p-4 relative overflow-hidden" style={{ background: T.card, border: `1px solid rgba(245,158,11,0.25)` }}>
                  <div className="absolute -top-4 -right-4 w-24 h-24 rounded-full pointer-events-none" style={{ background: 'rgba(245,158,11,0.05)' }} />
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-3.5 h-3.5" style={{ color: T.amber }} />
                    <p className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: T.amber }}>AI Insight</p>
                  </div>
                  <div className="flex items-baseline gap-2 mb-3">
                    <p className="text-2xl font-bold">{symbol}</p>
                    {signal && <SignalBadge action={signal.decision.action} />}
                  </div>

                  {prediction || signal ? (
                    <>
                      <div className="space-y-1.5 mb-3 text-xs">
                        <div className="flex justify-between items-center">
                          <span style={{ color: T.textMuted }}>Current Price</span>
                          <span className="font-semibold">{fmtMoney(currentPrice)}</span>
                        </div>
                        {prediction && (
                          <div className="flex justify-between items-center">
                            <span style={{ color: T.textMuted }}>Next-Day Forecast</span>
                            <span className="font-semibold" style={{ color: T.amber }}>{prediction.predicted_price != null ? fmtMoney(prediction.predicted_price) : 'N/A'}</span>
                          </div>
                        )}
                      </div>
                      {signal && <p className="text-[10px] leading-relaxed" style={{ color: T.textMuted }}>{signal.decision.reason}</p>}
                    </>
                  ) : (
                    <p className="text-[11px] mb-3" style={{ color: T.textMuted }}>Run the model on {symbol} to see a next-day forecast and buy/sell signal.</p>
                  )}

                  <button onClick={handleAnalyze} disabled={loading || currentPrice === 0}
                    className="w-full mt-3 py-2 rounded font-bold text-[11px] disabled:opacity-50 transition" style={{ background: T.amber, color: '#181000' }}>
                    {loading ? 'Analyzing…' : 'Run AI Analysis'}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-[10px] uppercase tracking-widest" style={{ color: T.textMuted }}>Holdings</p>
                    <button onClick={() => setActiveTab('portfolio')} className="text-[11px] hover:underline" style={{ color: T.primary }}>View all</button>
                  </div>
                  <div className="space-y-2.5">
                    {Object.entries(holdingsMap).slice(0, 5).map(([sym, h]) => (
                      <div key={sym} className="flex items-center gap-3">
                        <TickerBadge ticker={sym} />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold">{sym}</p>
                          <p className="text-[10px]" style={{ color: T.textMuted }}>{h.shares.toFixed(4)} shares</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs font-medium">{fmtMoney(h.currentValue)}</p>
                          <p className="text-[10px]" style={{ color: h.profitLoss >= 0 ? T.primary : T.red }}>{h.profitLoss >= 0 ? '+' : ''}{fmtMoney(h.profitLoss)}</p>
                        </div>
                      </div>
                    ))}
                    {holdingsCount === 0 && <p className="text-[11px] text-center py-6" style={{ color: T.textMuted }}>No holdings yet</p>}
                  </div>
                </div>

                <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Flame className="w-3.5 h-3.5" style={{ color: '#fb923c' }} />
                      <p className="text-[10px] uppercase tracking-widest" style={{ color: T.textMuted }}>Watchlist</p>
                    </div>
                    <button onClick={() => setActiveTab('watchlist')} className="text-[11px] hover:underline" style={{ color: T.primary }}>View all</button>
                  </div>
                  <div className="space-y-2">
                    {watchlist.slice(0, 5).map(sym => {
                      const q = stockQuotes[sym];
                      const stock = STOCK_DATABASE.find(s => s.symbol === sym);
                      return (
                        <div key={sym} className="flex items-center gap-3 cursor-pointer" onClick={() => jumpToSymbol(sym)}>
                          <span className="text-xs font-bold w-12 flex-shrink-0">{sym}</span>
                          <p className="flex-1 text-[10px] truncate" style={{ color: T.textMuted }}>{stock?.sector || ''}</p>
                          {q && <MiniTrend changePct={q.changePct} />}
                          <span className="text-xs font-bold w-16 text-right flex-shrink-0" style={{ color: q && q.changePct >= 0 ? T.primary : T.red }}>
                            {q ? fmtPct(q.changePct) : '—'}
                          </span>
                        </div>
                      );
                    })}
                    {watchlist.length === 0 && <p className="text-[11px] text-center py-6" style={{ color: T.textMuted }}>Your watchlist is empty</p>}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ─── ANALYSIS ─── */}
          {activeTab === 'analysis' && (
            <div className="p-5 space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                  <h1 className="text-lg font-bold">Analysis</h1>
                  <p className="text-[10px]" style={{ color: T.textMuted }}>Live price history, AI prediction &amp; manual trading</p>
                </div>
                <div className="flex items-center gap-1.5 flex-wrap">
                  {TRENDING_SYMBOLS.map(sym => (
                    <button key={sym} onClick={() => jumpToSymbol(sym)}
                      className="px-2.5 py-1 rounded text-[11px] transition-colors"
                      style={symbol === sym ? { background: T.primary, color: T.primaryFg } : { background: T.card, border: `1px solid ${T.border}`, color: T.textMuted }}>
                      {sym}
                    </button>
                  ))}
                </div>
              </div>

              {/* Stock info strip */}
              <div className="flex items-center gap-5 flex-wrap rounded-lg px-5 py-3" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                <div>
                  <p className="text-base font-bold">{symbol}</p>
                  <p className="text-[10px]" style={{ color: T.textMuted }}>{STOCK_DATABASE.find(s => s.symbol === symbol)?.name || 'Unknown'}</p>
                </div>
                <p className="text-xl font-bold">{fmtMoney(currentPrice)}</p>
                <p className="text-sm font-semibold" style={{ color: priceChange >= 0 ? T.primary : T.red }}>{fmtPct(priceChange)}</p>
                <div className="ml-auto flex items-center gap-4 text-[11px] flex-wrap">
                  <span style={{ color: T.textMuted }}>Vol: <span style={{ color: T.text }}>{latestVolume ? `${(latestVolume / 1000000).toFixed(2)}M` : '—'}</span></span>
                  <span style={{ color: T.textMuted }}>Sector: <span style={{ color: T.text }}>{STOCK_DATABASE.find(s => s.symbol === symbol)?.sector || '—'}</span></span>
                  <span style={{ color: T.textMuted }}>Your shares: <span style={{ color: T.text }}>{currentSymbolShares.toFixed(4)}</span></span>
                </div>
                <button onClick={() => toggleWatchlist(symbol)}
                  className="px-3 py-1.5 rounded text-[11px] font-semibold transition"
                  style={watchlist.includes(symbol) ? { background: 'rgba(0,229,153,0.1)', color: T.primary, border: `1px solid rgba(0,229,153,0.3)` } : { background: T.cardAlt, color: T.textMuted, border: `1px solid ${T.border}` }}>
                  {watchlist.includes(symbol) ? '★ Watching' : '+ Watch'}
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Chart column */}
                <div className="lg:col-span-2 space-y-4">
                  <div className="rounded-lg p-5" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                    <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
                      <div className="flex items-center gap-2 flex-wrap">
                        {TIME_RANGES.map(range => (
                          <button key={range.label} onClick={() => { setSelectedTimeRange(range.label); setUseCustomRange(false); }}
                            className="px-2.5 py-1 rounded text-[11px] transition-colors"
                            style={selectedTimeRange === range.label && !useCustomRange ? { background: 'rgba(0,229,153,0.1)', color: T.primary } : { color: T.textMuted }}>
                            {range.label}
                          </button>
                        ))}
                      </div>
                      <div className="flex gap-1">
                        {['area', 'line', 'bar', 'candlestick'].map((type) => (
                          <button key={type} onClick={() => setChartType(type)} className="px-2.5 py-1 rounded text-[11px] capitalize transition-colors"
                            style={chartType === type ? { background: T.primary, color: T.primaryFg } : { background: T.cardAlt, color: T.textMuted }}>
                            {type}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center gap-2 mb-2">
                      <input type="checkbox" id="customRange" checked={useCustomRange} onChange={(e) => setUseCustomRange(e.target.checked)} className="w-3 h-3" />
                      <label htmlFor="customRange" className="text-[10px]" style={{ color: T.textMuted }}>Custom date range</label>
                      {useCustomRange && (
                        <div className="flex gap-2 ml-2">
                          <input type="date" value={customStartDate} onChange={(e) => setCustomStartDate(e.target.value)}
                            className="qt-input rounded px-2 py-1 text-[10px]" style={{ background: T.cardAlt, border: `1px solid ${T.border}`, color: T.text }} />
                          <input type="date" value={customEndDate} onChange={(e) => setCustomEndDate(e.target.value)}
                            className="qt-input rounded px-2 py-1 text-[10px]" style={{ background: T.cardAlt, border: `1px solid ${T.border}`, color: T.text }} />
                        </div>
                      )}
                    </div>

                    {/* Zoom control — drag either handle to narrow the range
                        every chart type below (area/line/bar/candlestick)
                        reads from. */}
                    {filteredChartData.length > 20 && (
                      <div className="mb-4 rounded-lg px-3 pt-2.5 pb-2" style={{ background: T.cardAlt, border: `1px solid ${T.border}` }}>
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-[10px]" style={{ color: T.textMuted }}>
                            {zoomRange && filteredChartData[zoomRange[0]] && filteredChartData[zoomRange[1]]
                              ? `Zoomed: ${filteredChartData[zoomRange[0]].date} – ${filteredChartData[zoomRange[1]].date}`
                              : 'Drag the handles below to zoom in on a date range'}
                          </span>
                          {zoomRange && (
                            <button onClick={() => setZoomRange(null)} className="text-[10px] font-semibold flex-shrink-0" style={{ color: T.primary }}>
                              Reset zoom
                            </button>
                          )}
                        </div>
                        <ResponsiveContainer width="100%" height={44}>
                          <AreaChart data={filteredChartData} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
                            <Area type="monotone" dataKey="price" stroke={T.primary} strokeWidth={1} fill={T.primary} fillOpacity={0.12} />
                            <Brush
                              dataKey="date"
                              height={26}
                              stroke={T.primary}
                              fill="rgba(0,229,153,0.06)"
                              travellerWidth={12}
                              onChange={handleBrushChange}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {pricesLoading ? (
                      <div className="h-[260px] flex items-center justify-center">
                        <div className="text-center">
                          <div className="animate-spin h-8 w-8 border-2 rounded-full mx-auto mb-3" style={{ borderColor: T.border, borderTopColor: T.primary }}></div>
                          <p className="text-xs" style={{ color: T.textMuted }}>Loading {symbol} data…</p>
                        </div>
                      </div>
                    ) : prices.length === 0 ? (
                      <div className="h-[260px] flex items-center justify-center text-xs" style={{ color: T.textMuted }}>No price data for {symbol}</div>
                    ) : renderMainChart(260)}

                    <div className="flex items-center justify-between text-[10px] mt-2" style={{ color: T.textMuted }}>
                      <span>{zoomedChartData.length} of {filteredChartData.length} data points{zoomRange ? ' (zoomed)' : ''}</span>
                      {filteredYearRange && <span>{filteredYearRange}</span>}
                    </div>
                  </div>

                  {/* Volume */}
                  {prices.length > 0 && (
                    <div className="rounded-lg p-5" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                      <p className="text-[10px] uppercase tracking-widest mb-1" style={{ color: T.textMuted }}>Trading Volume</p>
                      <p className="text-[10px] mb-3" style={{ color: T.textMuted }}>
                        Shares traded each day. A price move on heavy volume shows real conviction — the same move on thin volume is easier to dismiss.
                      </p>
                      <ResponsiveContainer width="100%" height={120}>
                        <BarChart data={zoomedChartData} margin={{ top: 0, right: 20, left: 0, bottom: 0 }}>
                          <XAxis dataKey="date" tick={{ fontSize: 9, fill: T.textMuted }} axisLine={false} tickLine={false} interval={xAxisTickInterval} />
                          <YAxis tick={{ fontSize: 9, fill: T.textMuted }} axisLine={false} tickLine={false} tickFormatter={(v) => `${(v / 1000000).toFixed(0)}M`} width={36} />
                          <Tooltip content={<CustomTooltip />} />
                          <Bar dataKey="volume" fill={T.blue} fillOpacity={0.45} radius={[2, 2, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* AI summary */}
                  {(signal || prediction) && (
                    <div className="rounded-lg p-4" style={{ background: 'rgba(245,158,11,0.05)', border: `1px solid rgba(245,158,11,0.2)` }}>
                      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                        <div className="flex items-center gap-2">
                          <Brain className="w-4 h-4" style={{ color: T.amber }} />
                          <p className="text-xs font-semibold" style={{ color: T.amber }}>AI Analysis · {symbol}</p>
                          {signal && <SignalBadge action={signal.decision.action} />}
                        </div>
                        {signal && (
                          <button onClick={() => setShowEmailModal(true)} className="text-[11px] px-3 py-1 rounded font-semibold transition"
                            style={{ background: 'rgba(0,229,153,0.1)', color: T.primary, border: `1px solid rgba(0,229,153,0.25)` }}>
                            Email this signal
                          </button>
                        )}
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-3">
                        <div>
                          <p className="text-[10px] mb-0.5" style={{ color: T.textMuted }}>Current Price</p>
                          <p className="text-sm font-bold">{fmtMoney(currentPrice)}</p>
                        </div>
                        {prediction && (
                          <div>
                            <p className="text-[10px] mb-0.5" style={{ color: T.textMuted }}>Next-Day Forecast</p>
                            <p className="text-sm font-bold" style={{ color: T.amber }}>{prediction.predicted_price != null ? fmtMoney(prediction.predicted_price) : 'N/A'}</p>
                          </div>
                        )}
                      </div>
                      {signal && <p className="text-xs leading-relaxed">{signal.decision.reason}</p>}
                    </div>
                  )}

                  {/* Prediction accuracy — how past forecasts for this
                      symbol compared to what actually happened, once the
                      daily sync has caught up to the predicted date. */}
                  {!accuracyLoading && accuracy && accuracy.count > 0 && (
                    <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                      <div className="flex items-center gap-2 mb-3">
                        <Activity className="w-3.5 h-3.5" style={{ color: T.primary }} />
                        <p className="text-xs font-semibold">Prediction Track Record · {symbol}</p>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-1">
                        <div>
                          <p className="text-[10px] mb-0.5" style={{ color: T.textMuted }}>Avg. Price Error</p>
                          <p className="text-sm font-bold">{accuracy.avgAbsPctError != null ? `${accuracy.avgAbsPctError}%` : '—'}</p>
                        </div>
                        <div>
                          <p className="text-[10px] mb-0.5" style={{ color: T.textMuted }}>Direction Accuracy</p>
                          <p className="text-sm font-bold" style={{ color: accuracy.directionAccuracy >= 50 ? T.primary : T.red }}>
                            {accuracy.directionAccuracy != null ? `${accuracy.directionAccuracy}%` : '—'}
                          </p>
                        </div>
                        <div>
                          <p className="text-[10px] mb-0.5" style={{ color: T.textMuted }}>Scored Predictions</p>
                          <p className="text-sm font-bold">{accuracy.count}</p>
                        </div>
                      </div>
                      <p className="text-[10px] mt-2 mb-3" style={{ color: T.textMuted }}>
                        Direction accuracy is how often the model correctly called up-vs-down from the prior close — a more honest signal-quality check than price error alone.
                      </p>
                      {accuracy.history.length > 1 && (
                        <div>
                          <ResponsiveContainer width="100%" height={150}>
                            <LineChart
                              data={[...accuracy.history].reverse().map(h => ({
                                date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                                predicted: h.predicted,
                                actual: h.actual,
                              }))}
                              margin={{ top: 5, right: 10, left: 0, bottom: 0 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                              <XAxis dataKey="date" tick={{ fontSize: 9, fill: T.textMuted }} axisLine={false} tickLine={false} />
                              <YAxis tick={{ fontSize: 9, fill: T.textMuted }} axisLine={false} tickLine={false} width={42} domain={['auto', 'auto']} tickFormatter={(v) => `$${v.toFixed(0)}`} />
                              <Tooltip
                                contentStyle={{ background: T.card, border: `1px solid ${T.border}`, fontSize: 11, borderRadius: 6 }}
                                formatter={(value, name) => [fmtMoney(value), name === 'predicted' ? 'Predicted' : 'Actual']}
                              />
                              <Line type="monotone" dataKey="actual" stroke={T.primary} strokeWidth={2} dot={{ r: 2 }} name="actual" />
                              <Line type="monotone" dataKey="predicted" stroke={T.amber} strokeWidth={2} strokeDasharray="4 3" dot={{ r: 2 }} name="predicted" />
                            </LineChart>
                          </ResponsiveContainer>
                          <div className="flex items-center gap-4 mt-1 text-[10px]" style={{ color: T.textMuted }}>
                            <span className="flex items-center gap-1.5">
                              <span className="w-3 h-0.5 inline-block rounded-full" style={{ background: T.primary }} /> Actual close
                            </span>
                            <span className="flex items-center gap-1.5">
                              <span className="w-3 h-0.5 inline-block rounded-full" style={{ background: T.amber, backgroundImage: `repeating-linear-gradient(90deg, ${T.amber} 0 3px, transparent 3px 6px)` }} /> Predicted
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  {!accuracyLoading && (!accuracy || accuracy.count === 0) && (
                    <div className="rounded-lg p-4 text-[11px]" style={{ background: T.card, border: `1px solid ${T.border}`, color: T.textMuted }}>
                      No scored predictions for {symbol} yet — run AI Analysis on a few different days (with the daily price sync running) and this fills in once actual closes catch up to what was predicted.
                    </div>
                  )}
                </div>

                {/* Right column — trade + model status + trending */}
                <div className="space-y-4">
                  <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-[10px] uppercase tracking-widest" style={{ color: T.textMuted }}>AI Model</p>
                      <ModelStatusBadge modelInfo={modelInfo} />
                    </div>
                    <button onClick={handleAnalyze} disabled={loading || currentPrice === 0}
                      className="w-full py-2.5 rounded font-bold text-xs disabled:opacity-50 transition" style={{ background: T.amber, color: '#181000' }}>
                      {loading ? 'Analyzing…' : 'Run AI Analysis'}
                    </button>
                  </div>

                  <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                    <p className="text-[10px] uppercase tracking-widest mb-3" style={{ color: T.textMuted }}>Manual Trading</p>
                    <label className="text-[10px] mb-1.5 block" style={{ color: T.textMuted }}>Shares to trade</label>
                    <input type="number" step="0.0001" value={manualTradeAmount} onChange={(e) => setManualTradeAmount(e.target.value)}
                      placeholder="Enter shares…" min="0" className="qt-input w-full rounded px-3 py-2 text-xs mb-1"
                      style={{ background: T.cardAlt, border: `1px solid ${T.border}`, color: T.text }} />
                    {manualTradeAmount && !isNaN(parseFloat(manualTradeAmount)) && (
                      <p className="text-[10px] mb-3" style={{ color: T.textMuted }}>Cost: {fmtMoney(parseFloat(manualTradeAmount) * currentPrice)}</p>
                    )}
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <button onClick={() => executeTrade('BUY')} disabled={!manualTradeAmount || currentPrice === 0 || tradeSubmitting}
                        className="py-2 rounded font-bold text-xs transition disabled:opacity-50" style={{ background: T.primary, color: T.primaryFg }}>
                        Buy
                      </button>
                      <button onClick={() => executeTrade('SELL')} disabled={!manualTradeAmount || currentPrice === 0 || tradeSubmitting}
                        className="py-2 rounded font-bold text-xs transition disabled:opacity-50" style={{ background: T.red, color: '#fff' }}>
                        Sell
                      </button>
                    </div>
                    <p className="text-[10px] mt-3 text-center" style={{ color: T.textMuted }}>Holdings: {currentSymbolShares.toFixed(4)} shares</p>
                  </div>

                  <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                    <p className="text-[10px] uppercase tracking-widest mb-3" style={{ color: T.textMuted }}>Trending</p>
                    <div className="space-y-1">
                      {TRENDING_SYMBOLS.map(sym => {
                        const stock = STOCK_DATABASE.find(s => s.symbol === sym);
                        const q = stockQuotes[sym];
                        const isSelected = sym === symbol;
                        return (
                          <div key={sym} className="rounded transition"
                            style={isSelected ? { background: 'rgba(0,229,153,0.08)', border: `1px solid rgba(0,229,153,0.2)` } : { border: '1px solid transparent' }}>
                            <div className="flex items-center p-2 gap-2">
                              <button onClick={() => jumpToSymbol(sym)} className="flex-1 text-left flex items-center gap-2 min-w-0">
                                <TickerBadge ticker={sym} />
                                <div className="min-w-0">
                                  <p className="text-xs font-bold">{sym}</p>
                                  <p className="text-[10px] truncate" style={{ color: T.textMuted }}>{stock?.name}</p>
                                </div>
                              </button>
                              {q && <span className="text-[11px] font-semibold flex-shrink-0" style={{ color: q.changePct >= 0 ? T.primary : T.red }}>{fmtPct(q.changePct)}</span>}
                              <button onClick={() => toggleWatchlist(sym)} className="p-1 flex-shrink-0" style={{ color: watchlist.includes(sym) ? T.primary : T.textMuted }}>★</button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ─── WATCHLIST ─── */}
          {activeTab === 'watchlist' && (
            <div className="p-5 space-y-5">
              <div>
                <h1 className="text-lg font-bold">Watchlist</h1>
                <p className="text-[10px]" style={{ color: T.textMuted }}>{watchlist.length} stock{watchlist.length === 1 ? '' : 's'} monitored</p>
              </div>

              <div className="rounded-lg overflow-hidden" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                {watchlist.length === 0 ? (
                  <p className="text-center py-16 text-xs" style={{ color: T.textMuted }}>No stocks in watchlist yet — add some from the Analysis tab.</p>
                ) : (
                  <div className="overflow-x-auto">
                  <table className="w-full min-w-[720px]">
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                        {['Symbol', 'Company', 'Price', 'Change', 'Sector', 'Trend', ''].map(h => (
                          <th key={h} className="px-4 py-2.5 text-left text-[9px] uppercase tracking-widest font-medium whitespace-nowrap" style={{ color: T.textMuted }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {watchlist.map((sym, i) => {
                        const stock = STOCK_DATABASE.find(s => s.symbol === sym);
                        const q = stockQuotes[sym];
                        return (
                          <tr key={sym} style={{ borderBottom: `1px solid ${T.border}`, background: i % 2 ? 'rgba(255,255,255,0.015)' : 'transparent' }}>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <TickerBadge ticker={sym} />
                                <span className="text-xs font-bold">{sym}</span>
                              </div>
                            </td>
                            <td className="px-4 py-3 text-[11px]" style={{ color: T.textMuted }}>{stock?.name || 'Unknown'}</td>
                            <td className="px-4 py-3 text-xs font-semibold">{q ? fmtMoney(q.price) : '—'}</td>
                            <td className="px-4 py-3">
                              {q ? <span className="text-xs font-semibold" style={{ color: q.changePct >= 0 ? T.primary : T.red }}>{fmtPct(q.changePct)}</span> : <span className="text-xs" style={{ color: T.textMuted }}>—</span>}
                            </td>
                            <td className="px-4 py-3 text-[11px]" style={{ color: T.textMuted }}>{stock?.sector || '—'}</td>
                            <td className="px-4 py-3">{q ? <MiniTrend changePct={q.changePct} /> : <span className="text-[10px]" style={{ color: T.textMuted }}>—</span>}</td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2 justify-end">
                                <button onClick={() => jumpToSymbol(sym)} className="text-[11px] px-2.5 py-1 rounded font-semibold transition" style={{ background: T.primary, color: T.primaryFg }}>Analyze</button>
                                <button onClick={() => toggleWatchlist(sym)} style={{ color: T.red }} title="Remove"><X className="w-3.5 h-3.5" /></button>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ─── PORTFOLIO ─── */}
          {activeTab === 'portfolio' && (
            <div className="p-5 space-y-5">
              <div>
                <h1 className="text-lg font-bold">Portfolio</h1>
                <p className="text-[10px]" style={{ color: T.textMuted }}>Server-verified holdings overview</p>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <StatCard label="Total Value" value={fmtMoney(totalValue)} sub="Live pricing" />
                <StatCard label="Cash Balance" value={fmtMoney(balance)} sub={totalValue ? `${((balance / totalValue) * 100).toFixed(1)}% of portfolio` : undefined} />
                <StatCard label="Total P/L" value={`${totalProfitLoss >= 0 ? '+' : ''}${fmtMoney(totalProfitLoss)}`}
                  sub={portfolio?.initialBalance ? fmtPct((totalProfitLoss / portfolio.initialBalance) * 100) + ' all-time' : undefined} positive={totalProfitLoss >= 0} />
                <StatCard label="Positions" value={`${holdingsCount} stock${holdingsCount === 1 ? '' : 's'}`} />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="lg:col-span-2 rounded-lg overflow-hidden" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                  <div className="px-4 py-3" style={{ borderBottom: `1px solid ${T.border}` }}>
                    <p className="text-[10px] uppercase tracking-widest" style={{ color: T.textMuted }}>Holdings</p>
                  </div>
                  {holdingsCount === 0 ? (
                    <p className="text-center py-16 text-xs" style={{ color: T.textMuted }}>No holdings yet</p>
                  ) : (
                    <div className="overflow-x-auto">
                    <table className="w-full min-w-[640px]">
                      <thead>
                        <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                          {['Stock', 'Shares', 'Current', 'Mkt Value', 'Gain/Loss', 'Return'].map(h => (
                            <th key={h} className="px-4 py-2.5 text-left text-[9px] uppercase tracking-widest font-medium" style={{ color: T.textMuted }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(holdingsMap).map(([sym, h]) => {
                          const priceNow = h.shares > 0 ? h.currentValue / h.shares : 0;
                          const cost = h.currentValue - h.profitLoss;
                          const pct = cost ? (h.profitLoss / cost) * 100 : 0;
                          const stock = STOCK_DATABASE.find(s => s.symbol === sym);
                          return (
                            <tr key={sym} style={{ borderBottom: `1px solid ${T.border}` }}>
                              <td className="px-4 py-3">
                                <div className="flex items-center gap-2">
                                  <TickerBadge ticker={sym} />
                                  <div>
                                    <p className="text-xs font-bold">{sym}</p>
                                    <p className="text-[9px]" style={{ color: T.textMuted }}>{stock?.name}</p>
                                  </div>
                                </div>
                              </td>
                              <td className="px-4 py-3 text-xs">{h.shares.toFixed(4)}</td>
                              <td className="px-4 py-3 text-xs">{fmtMoney(priceNow)}</td>
                              <td className="px-4 py-3 text-xs font-semibold">{fmtMoney(h.currentValue)}</td>
                              <td className="px-4 py-3 text-xs font-semibold" style={{ color: h.profitLoss >= 0 ? T.primary : T.red }}>{h.profitLoss >= 0 ? '+' : ''}{fmtMoney(h.profitLoss)}</td>
                              <td className="px-4 py-3"><span className="text-xs font-bold" style={{ color: pct >= 0 ? T.primary : T.red }}>{fmtPct(pct)}</span></td>
                            </tr>
                          );
                        })}
                        <tr>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <div className="w-8 h-8 rounded-md flex items-center justify-center text-[10px] font-bold" style={{ background: T.muted, color: T.textMuted }}>$</div>
                              <div>
                                <p className="text-xs font-bold">CASH</p>
                                <p className="text-[9px]" style={{ color: T.textMuted }}>USD Balance</p>
                              </div>
                            </div>
                          </td>
                          {['—', '$1.00', fmtMoney(balance), '—', '—'].map((v, i) => (
                            <td key={i} className="px-4 py-3 text-xs" style={{ color: T.textMuted }}>{v}</td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                    </div>
                  )}
                </div>

                <div className="rounded-lg p-4" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                  <p className="text-[10px] uppercase tracking-widest mb-3" style={{ color: T.textMuted }}>Allocation</p>
                  {totalValue > 0 ? (
                    <>
                      <ResponsiveContainer width="100%" height={170}>
                        <PieChart>
                          <Pie data={pieData} cx="50%" cy="50%" innerRadius={48} outerRadius={75} dataKey="value" paddingAngle={3}>
                            {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                          </Pie>
                          <Tooltip formatter={(v) => fmtMoney(v)} contentStyle={{ background: T.card, border: `1px solid ${T.border}`, fontSize: 11 }} />
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="space-y-1.5 mt-2">
                        {pieData.map((d, i) => (
                          <div key={d.name} className="flex items-center justify-between text-[11px]">
                            <div className="flex items-center gap-2">
                              <span className="w-2 h-2 rounded-sm flex-shrink-0" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                              <span style={{ color: T.textMuted }}>{d.name}</span>
                            </div>
                            <span className="font-medium">{totalValue ? ((d.value / totalValue) * 100).toFixed(1) : '0.0'}%</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <p className="text-center py-10 text-xs" style={{ color: T.textMuted }}>No allocation data</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ─── HISTORY ─── */}
          {activeTab === 'history' && (
            <div className="p-5 space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                  <h1 className="text-lg font-bold">Trading History</h1>
                  <p className="text-[10px]" style={{ color: T.textMuted }}>{tradeHistory.length} trade{tradeHistory.length === 1 ? '' : 's'} logged</p>
                </div>
                <button onClick={exportTradeHistory} disabled={tradeHistory.length === 0}
                  className="px-4 py-2 rounded font-semibold text-xs disabled:opacity-50 transition" style={{ background: T.primary, color: T.primaryFg }}>
                  Export CSV
                </button>
              </div>

              <div className="rounded-lg overflow-hidden" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                {tradeHistory.length === 0 ? (
                  <p className="text-center py-16 text-xs" style={{ color: T.textMuted }}>No trading history yet</p>
                ) : (
                  <div className="overflow-x-auto">
                  <table className="w-full min-w-[640px]">
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                        {['Date', 'Symbol', 'Action', 'Price', 'Quantity', 'P/L'].map(h => (
                          <th key={h} className="px-4 py-2.5 text-left text-[9px] uppercase tracking-widest font-medium" style={{ color: T.textMuted }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {tradeHistory.map((trade, i) => (
                        <tr key={trade.id} style={{ borderBottom: `1px solid ${T.border}`, background: i % 2 ? 'rgba(255,255,255,0.015)' : 'transparent' }}>
                          <td className="px-4 py-3 text-[11px]" style={{ color: T.textMuted }}>{new Date(trade.timestamp).toLocaleString()}</td>
                          <td className="px-4 py-3 text-xs font-bold">{trade.symbol}</td>
                          <td className="px-4 py-3">
                            <span className="text-[10px] px-2 py-0.5 rounded-full font-bold" style={{ background: trade.action === 'BUY' ? 'rgba(0,229,153,0.12)' : 'rgba(239,68,68,0.12)', color: trade.action === 'BUY' ? T.primary : T.red }}>
                              {trade.action}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-xs font-semibold">{fmtMoney(trade.price)}</td>
                          <td className="px-4 py-3 text-xs" style={{ color: T.textMuted }}>{parseFloat(trade.quantity).toFixed(4)}</td>
                          <td className="px-4 py-3 text-xs font-semibold" style={{ color: trade.profit_loss == null ? T.textMuted : parseFloat(trade.profit_loss) >= 0 ? T.primary : T.red }}>
                            {trade.profit_loss != null ? `${parseFloat(trade.profit_loss) >= 0 ? '+' : ''}${fmtMoney(trade.profit_loss)}` : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ─── ABOUT ─── */}
          {activeTab === 'about' && (
            <div className="p-5">
              <div className="max-w-2xl mx-auto rounded-lg p-8" style={{ background: T.card, border: `1px solid ${T.border}` }}>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: T.primary }}>
                    <Brain className="w-5 h-5" style={{ color: T.primaryFg }} />
                  </div>
                  <h1 className="text-2xl font-bold">Quantum Trade</h1>
                </div>
                <p className="text-sm mb-8" style={{ color: T.textMuted }}>
                  AI-assisted trading platform covering {STOCK_DATABASE.length} stocks, logged in as {user?.email}.
                </p>

                {modelInfo?.lstm_loaded ? (
                  <div className="rounded-lg p-5 mb-5" style={{ background: 'rgba(0,229,153,0.05)', border: `1px solid rgba(0,229,153,0.2)` }}>
                    <h3 className="text-sm font-bold mb-3">LSTM Model Active</h3>
                    <div className="space-y-1.5 text-xs" style={{ color: T.textMuted }}>
                      {modelInfo.metadata?.total_samples != null && <p>Training samples: <span style={{ color: T.text }}>{modelInfo.metadata.total_samples.toLocaleString()}</span></p>}
                      {modelInfo.metadata?.sequence_length != null && <p>Lookback window: <span style={{ color: T.text }}>{modelInfo.metadata.sequence_length} days</span></p>}
                      {modelInfo.metadata?.num_features != null && <p>Input features: <span style={{ color: T.text }}>{modelInfo.metadata.num_features}</span></p>}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-lg p-5 mb-5" style={{ background: 'rgba(245,158,11,0.05)', border: `1px solid rgba(245,158,11,0.2)` }}>
                    <h3 className="text-sm font-bold">Prediction model unavailable</h3>
                    <p className="text-xs mt-2" style={{ color: T.textMuted }}>The LSTM model isn't currently loaded on the ML service.</p>
                  </div>
                )}

                <div className="rounded-lg p-5 mb-5" style={{ background: T.cardAlt, border: `1px solid ${T.border}` }}>
                  <h3 className="text-sm font-bold mb-3">Platform</h3>
                  <div className="space-y-2 text-xs" style={{ color: T.textMuted }}>
                    <p><strong style={{ color: T.text }}>Account login:</strong> JWT-based authentication, portfolios persisted server-side per account.</p>
                    <p><strong style={{ color: T.text }}>Audit logging:</strong> every trade and prediction request is logged server-side.</p>
                    <p><strong style={{ color: T.text }}>Trade execution:</strong> prices are fetched and validated server-side, never trusted from the client.</p>
                  </div>
                </div>

                <div className="rounded-lg p-5" style={{ background: 'rgba(245,158,11,0.05)', border: `1px solid rgba(245,158,11,0.2)` }}>
                  <h3 className="text-sm font-bold mb-1">Disclaimer</h3>
                  <p className="text-xs" style={{ color: T.textMuted }}>Educational platform only. Not financial advice.</p>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      <EmailAlertModal isOpen={showEmailModal} onClose={() => setShowEmailModal(false)} symbol={symbol} signal={signal} />
    </div>
  );
}

function AppContent() {
  const { token } = useAuth();
  return token ? <MainApp /> : <AuthScreen />;
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;