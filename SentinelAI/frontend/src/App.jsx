import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  ShieldAlert, Activity, ShieldCheck, AlertTriangle, 
  Zap, ReceiptText, Share2, Cpu, FileSpreadsheet,
  CheckCircle, Play, Sliders, RefreshCw, Search,
  ArrowRight, Filter, Download, Info, Check, X
} from 'lucide-react';

export default function App() {
  const [tab, setTab] = useState('overview');
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [presets, setPresets] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState(null);

  const [simForm, setSimForm] = useState({
    type: 'TRANSFER',
    amount: 185000.00,
    oldbalanceOrg: 185000.00,
    newbalanceOrig: 0.00,
    oldbalanceDest: 12000.00,
    newbalanceDest: 12000.00,
    nameOrig: 'C891273941',
    nameDest: 'MULE_007',
    step: 142
  });
  const [predicting, setPredicting] = useState(false);
  const [simResult, setSimResult] = useState(null);

  const [transactions, setTransactions] = useState([]);
  const [ledgerFilter, setLedgerFilter] = useState('ALL');
  const [ledgerSearch, setLedgerSearch] = useState('');
  const [ledgerLoading, setLedgerLoading] = useState(false);

  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);

  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const canvasRef = useRef(null);

  const [batchResults, setBatchResults] = useState(null);
  const [batchProcessing, setBatchProcessing] = useState(false);

  useEffect(() => {
    fetchStats();
    fetchPresets();
  }, []);

  useEffect(() => {
    if (tab === 'ledger') fetchTransactions();
    if (tab === 'models') fetchMetrics();
    if (tab === 'network') fetchGraph();
  }, [tab]);

  const fetchStats = async () => {
    try {
      const res = await axios.get('/api/v1/stats');
      setStats(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const fetchPresets = async () => {
    try {
      const res = await axios.get('/api/v1/presets');
      setPresets(res.data);
      if (res.data && res.data.length > 0) {
        setSelectedPreset(res.data[0]);
        setSimForm(res.data[0].payload);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchTransactions = async () => {
    setLedgerLoading(true);
    try {
      const res = await axios.get('/api/v1/transactions?limit=50');
      setTransactions(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLedgerLoading(false);
    }
  };

  const fetchMetrics = async () => {
    setMetricsLoading(true);
    try {
      const res = await axios.get('/api/v1/metrics');
      setMetrics(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setMetricsLoading(false);
    }
  };

  const fetchGraph = async () => {
    setGraphLoading(true);
    try {
      const res = await axios.get('/api/v1/graph/subgraph');
      setGraphData(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setGraphLoading(false);
    }
  };

  useEffect(() => {
    if (!graphData || tab !== 'network' || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const nodes = graphData.nodes.map((n, i) => {
      const angle = (i / graphData.nodes.length) * 2 * Math.PI;
      const radius = n.type === 'mule' ? 120 : (n.type === 'hub' ? 180 : 250);
      return {
        ...n,
        x: width / 2 + radius * Math.cos(angle) + (Math.random() - 0.5) * 40,
        y: height / 2 + radius * Math.sin(angle) + (Math.random() - 0.5) * 40
      };
    });

    const nodeMap = new Map();
    nodes.forEach(n => nodeMap.set(n.id, n));

    ctx.lineWidth = 1;
    graphData.edges.forEach(edge => {
      const s = nodeMap.get(edge.source);
      const t = nodeMap.get(edge.target);
      if (s && t) {
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.strokeStyle = edge.is_fraud ? 'rgba(244, 63, 94, 0.4)' : 'rgba(75, 85, 99, 0.3)';
        ctx.stroke();
      }
    });

    nodes.forEach(n => {
      ctx.beginPath();
      const r = n.type === 'mule' ? 9 : (n.type === 'hub' ? 7 : 5);
      ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
      if (n.type === 'mule') {
        ctx.fillStyle = '#f43f5e';
      } else if (n.type === 'hub') {
        ctx.fillStyle = '#a855f7';
      } else {
        ctx.fillStyle = '#3b82f6';
      }
      ctx.fill();
      ctx.strokeStyle = '#0B0F17';
      ctx.lineWidth = 2;
      ctx.stroke();

      if (n.type === 'mule' || n.type === 'hub') {
        ctx.fillStyle = '#9ca3af';
        ctx.font = '10px sans-serif';
        ctx.fillText(n.id, n.x + 12, n.y + 3);
      }
    });
  }, [graphData, tab]);

  const handlePresetSelect = (preset) => {
    setSelectedPreset(preset);
    setSimForm(preset.payload);
  };

  const handleSimSubmit = async (e) => {
    e.preventDefault();
    setPredicting(true);
    try {
      const res = await axios.post('/api/v1/predict', simForm);
      setSimResult(res.data);
      fetchStats();
    } catch (err) {
      console.error(err);
    } finally {
      setPredicting(false);
    }
  };

  const handleRunBatchSample = async () => {
    setBatchProcessing(true);
    try {
      const sample = [
        { step: 100, type: 'TRANSFER', amount: 150000, nameOrig: 'C1001', oldbalanceOrg: 150000, newbalanceOrig: 0, nameDest: 'MULE_001', oldbalanceDest: 0, newbalanceDest: 0 },
        { step: 101, type: 'PAYMENT', amount: 45.5, nameOrig: 'C1002', oldbalanceOrg: 500, newbalanceOrig: 454.5, nameDest: 'M_HUB_01', oldbalanceDest: 1000, newbalanceDest: 1045.5 },
        { step: 102, type: 'TRANSFER', amount: 9800, nameOrig: 'C1003', oldbalanceOrg: 9800, newbalanceOrig: 0, nameDest: 'MULE_002', oldbalanceDest: 50000, newbalanceDest: 59800 },
        { step: 103, type: 'CASH_OUT', amount: 320000, nameOrig: 'C1004', oldbalanceOrg: 320000, newbalanceOrig: 0, nameDest: 'MULE_003', oldbalanceDest: 0, newbalanceDest: 0 },
        { step: 104, type: 'DEBIT', amount: 120, nameOrig: 'C1005', oldbalanceOrg: 2000, newbalanceOrig: 1880, nameDest: 'M_HUB_02', oldbalanceDest: 5000, newbalanceDest: 5120 }
      ];
      const res = await axios.post('/api/v1/predict/batch', sample);
      setBatchResults(res.data);
      fetchStats();
    } catch (e) {
      console.error(e);
    } finally {
      setBatchProcessing(false);
    }
  };

  const filteredTransactions = transactions.filter(tx => {
    if (ledgerFilter !== 'ALL' && tx.risk_tier !== ledgerFilter) return false;
    if (ledgerSearch) {
      const q = ledgerSearch.toLowerCase();
      return tx.nameOrig?.toLowerCase().includes(q) || tx.nameDest?.toLowerCase().includes(q) || tx.type?.toLowerCase().includes(q);
    }
    return true;
  });

  const getTierColor = (tier) => {
    switch (tier) {
      case 'CRITICAL': return 'bg-rose-500/20 text-rose-400 border-rose-500/30';
      case 'HIGH': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      case 'MEDIUM': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'LOW': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      default: return 'bg-gray-700 text-gray-300';
    }
  };

  const getRecColor = (rec) => {
    switch (rec) {
      case 'DECLINE': return 'bg-rose-600 text-white';
      case 'MANUAL_REVIEW': return 'bg-amber-600 text-white';
      case 'STEP_UP_AUTH': return 'bg-yellow-600 text-black';
      case 'APPROVE': return 'bg-emerald-600 text-white';
      default: return 'bg-gray-600 text-white';
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F17] text-gray-100 flex flex-col font-sans">
      <header className="sticky top-0 z-50 border-b border-gray-800 bg-[#0B0F17]/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <ShieldAlert className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight text-white">
                Sentinel<span className="text-emerald-400">AI</span>
              </span>
              <p className="text-xs text-gray-400">Intelligent Fraud & Risk Intelligence</p>
            </div>
          </div>

          <div className="flex items-center space-x-1 sm:space-x-2">
            {[
              { id: 'overview', label: 'Overview', icon: Activity },
              { id: 'simulator', label: 'Simulator', icon: Zap },
              { id: 'ledger', label: 'Ledger', icon: ReceiptText },
              { id: 'network', label: 'Mule Ring', icon: Share2 },
              { id: 'models', label: 'Models & XAI', icon: Cpu },
              { id: 'batch', label: 'Batch Studio', icon: FileSpreadsheet }
            ].map(t => {
              const Icon = t.icon;
              return (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 ${
                    tab === t.id ? 'bg-gray-800 text-emerald-400 border border-emerald-500/30' : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{t.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6 space-y-6">
        {loading ? (
          <div className="flex justify-center items-center h-64 text-gray-400">
            <RefreshCw className="w-6 h-6 animate-spin mr-2 text-emerald-400" />
            Loading SentinelAI Intelligence System...
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="rounded-2xl p-5 border border-gray-800 bg-[#111827]/70 backdrop-blur-md">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Total Volume Audited</span>
                <h3 className="text-2xl font-bold text-white mt-2">${stats?.total_volume_analyzed?.toLocaleString(undefined, { minimumFractionDigits: 2 })}</h3>
                <p className="text-xs text-gray-400 mt-1">{stats?.total_transactions?.toLocaleString()} live transactions</p>
              </div>

              <div className="rounded-2xl p-5 border border-gray-800 bg-[#111827]/70 backdrop-blur-md">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Prevented Fraud Loss</span>
                <h3 className="text-2xl font-bold text-emerald-400 mt-2">${stats?.prevented_loss_volume?.toLocaleString(undefined, { minimumFractionDigits: 2 })}</h3>
                <p className="text-xs text-gray-400 mt-1">Intercepted illicit capital</p>
              </div>

              <div className="rounded-2xl p-5 border border-gray-800 bg-[#111827]/70 backdrop-blur-md">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Fraud Flagged</span>
                <h3 className="text-2xl font-bold text-rose-400 mt-2">{stats?.fraud_detected_count?.toLocaleString()}</h3>
                <p className="text-xs text-rose-300/80 mt-1">{stats?.fraud_rate_pct}% of processed volume</p>
              </div>

              <div className="rounded-2xl p-5 border border-gray-800 bg-[#111827]/70 backdrop-blur-md">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Active Defense Ensemble</span>
                <h3 className="text-2xl font-bold text-purple-400 mt-2">5 Models Fused</h3>
                <p className="text-xs text-purple-300/80 mt-1">XGB, RF, GNN, LR, IsoForest</p>
              </div>
            </div>

            {tab === 'overview' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-lg font-bold text-white flex items-center gap-2">
                        <Activity className="w-5 h-5 text-emerald-400" />
                        Risk Distribution Breakdown
                      </h4>
                      <button onClick={fetchStats} className="text-xs text-gray-400 hover:text-white flex items-center gap-1">
                        <RefreshCw className="w-3.5 h-3.5" /> Refresh
                      </button>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
                      <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20">
                        <span className="text-xs text-rose-400 font-semibold uppercase">Critical (Decline)</span>
                        <div className="text-2xl font-bold text-rose-300 mt-1">{stats?.risk_distribution?.CRITICAL || 0}</div>
                        <span className="text-[11px] text-gray-400">Score &ge; 80%</span>
                      </div>
                      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
                        <span className="text-xs text-amber-400 font-semibold uppercase">High (Review)</span>
                        <div className="text-2xl font-bold text-amber-300 mt-1">{stats?.risk_distribution?.HIGH || 0}</div>
                        <span className="text-[11px] text-gray-400">Score 60 - 79%</span>
                      </div>
                      <div className="p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/20">
                        <span className="text-xs text-yellow-400 font-semibold uppercase">Medium (Auth)</span>
                        <div className="text-2xl font-bold text-yellow-300 mt-1">{stats?.risk_distribution?.MEDIUM || 0}</div>
                        <span className="text-[11px] text-gray-400">Score 35 - 59%</span>
                      </div>
                      <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                        <span className="text-xs text-emerald-400 font-semibold uppercase">Low (Approve)</span>
                        <div className="text-2xl font-bold text-emerald-300 mt-1">{stats?.risk_distribution?.LOW || 0}</div>
                        <span className="text-[11px] text-gray-400">Score &lt; 35%</span>
                      </div>
                    </div>

                    <div className="p-4 rounded-xl border border-gray-800 bg-[#0B0F17]/60 space-y-2">
                      <div className="text-xs font-semibold text-gray-400">Composite Risk Distribution Profile</div>
                      <div className="w-full bg-gray-800 h-3 rounded-full flex overflow-hidden">
                        <div style={{ width: `${(stats?.risk_distribution?.CRITICAL / (stats?.total_transactions || 1)) * 100}%` }} className="bg-rose-500"></div>
                        <div style={{ width: `${(stats?.risk_distribution?.HIGH / (stats?.total_transactions || 1)) * 100}%` }} className="bg-amber-500"></div>
                        <div style={{ width: `${(stats?.risk_distribution?.MEDIUM / (stats?.total_transactions || 1)) * 100}%` }} className="bg-yellow-500"></div>
                        <div style={{ width: `${(stats?.risk_distribution?.LOW / (stats?.total_transactions || 1)) * 100}%` }} className="bg-emerald-500"></div>
                      </div>
                      <div className="flex justify-between text-[11px] text-gray-400">
                        <span>Low ({stats?.risk_distribution?.LOW})</span>
                        <span>Med ({stats?.risk_distribution?.MEDIUM})</span>
                        <span>High ({stats?.risk_distribution?.HIGH})</span>
                        <span>Critical ({stats?.risk_distribution?.CRITICAL})</span>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-4">
                    <h4 className="text-lg font-bold text-white flex items-center gap-2">
                      <Zap className="w-5 h-5 text-amber-400" />
                      Quick Simulation Presets
                    </h4>
                    <p className="text-xs text-gray-400">Select an attack pattern to test model inference:</p>

                    <div className="space-y-2">
                      {presets.map((p, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            handlePresetSelect(p);
                            setTab('simulator');
                          }}
                          className="w-full text-left p-3 rounded-xl border border-gray-800 hover:border-emerald-500/40 bg-[#0B0F17]/50 hover:bg-[#0B0F17] transition group"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-sm text-gray-200 group-hover:text-emerald-400">{p.name}</span>
                            <ArrowRight className="w-4 h-4 text-gray-500 group-hover:text-emerald-400 transition" />
                          </div>
                          <p className="text-xs text-gray-400 mt-1 line-clamp-1">{p.description}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {tab === 'simulator' && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-5 space-y-4">
                  <div className="rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-4">
                    <div className="flex items-center justify-between border-b border-gray-800 pb-3">
                      <h4 className="font-bold text-white flex items-center gap-2">
                        <Sliders className="w-5 h-5 text-emerald-400" />
                        Transaction Parameters
                      </h4>
                      <span className="text-xs text-gray-400">Step {simForm.step}</span>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      {presets.map((p, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => handlePresetSelect(p)}
                          className={`text-xs px-2.5 py-1 rounded-md transition ${
                            selectedPreset?.name === p.name 
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' 
                              : 'bg-gray-800 text-gray-400 hover:text-white'
                          }`}
                        >
                          {p.name.split(' ')[0]}
                        </button>
                      ))}
                    </div>

                    <form onSubmit={handleSimSubmit} className="space-y-3">
                      <div>
                        <label className="text-xs font-semibold text-gray-400">Transaction Type</label>
                        <select
                          value={simForm.type}
                          onChange={e => setSimForm({ ...simForm, type: e.target.value })}
                          className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                        >
                          <option value="TRANSFER">TRANSFER</option>
                          <option value="CASH_OUT">CASH_OUT</option>
                          <option value="PAYMENT">PAYMENT</option>
                          <option value="DEBIT">DEBIT</option>
                          <option value="CASH_IN">CASH_IN</option>
                        </select>
                      </div>

                      <div>
                        <label className="text-xs font-semibold text-gray-400">Transaction Amount ($)</label>
                        <input
                          type="number"
                          step="0.01"
                          value={simForm.amount}
                          onChange={e => setSimForm({ ...simForm, amount: parseFloat(e.target.value) || 0 })}
                          className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-xs font-semibold text-gray-400">Origin Initial ($)</label>
                          <input
                            type="number"
                            step="0.01"
                            value={simForm.oldbalanceOrg}
                            onChange={e => setSimForm({ ...simForm, oldbalanceOrg: parseFloat(e.target.value) || 0 })}
                            className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                        <div>
                          <label className="text-xs font-semibold text-gray-400">Origin Final ($)</label>
                          <input
                            type="number"
                            step="0.01"
                            value={simForm.newbalanceOrig}
                            onChange={e => setSimForm({ ...simForm, newbalanceOrig: parseFloat(e.target.value) || 0 })}
                            className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-xs font-semibold text-gray-400">Dest Initial ($)</label>
                          <input
                            type="number"
                            step="0.01"
                            value={simForm.oldbalanceDest}
                            onChange={e => setSimForm({ ...simForm, oldbalanceDest: parseFloat(e.target.value) || 0 })}
                            className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                        <div>
                          <label className="text-xs font-semibold text-gray-400">Dest Final ($)</label>
                          <input
                            type="number"
                            step="0.01"
                            value={simForm.newbalanceDest}
                            onChange={e => setSimForm({ ...simForm, newbalanceDest: parseFloat(e.target.value) || 0 })}
                            className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-xs font-semibold text-gray-400">Origin Account</label>
                          <input
                            type="text"
                            value={simForm.nameOrig}
                            onChange={e => setSimForm({ ...simForm, nameOrig: e.target.value })}
                            className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                        <div>
                          <label className="text-xs font-semibold text-gray-400">Dest Account</label>
                          <input
                            type="text"
                            value={simForm.nameDest}
                            onChange={e => setSimForm({ ...simForm, nameDest: e.target.value })}
                            className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                      </div>

                      <button
                        type="submit"
                        disabled={predicting}
                        className="w-full mt-2 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-white font-semibold flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20 transition disabled:opacity-50"
                      >
                        {predicting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                        Run Multi-Model Fraud Audit
                      </button>
                    </form>
                  </div>
                </div>

                <div className="lg:col-span-7 space-y-4">
                  {simResult ? (
                    <div className="space-y-4">
                      <div className="rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Audit Result</span>
                            <div className="flex items-center gap-3 mt-1">
                              <span className={`text-2xl font-bold px-3 py-0.5 rounded-lg border ${getTierColor(simResult.risk_tier)}`}>
                                {simResult.risk_tier} RISK
                              </span>
                              <span className={`text-sm font-semibold px-3 py-1 rounded-md ${getRecColor(simResult.recommendation)}`}>
                                {simResult.recommendation}
                              </span>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className="text-xs text-gray-400">Composite Risk Index</span>
                            <div className="text-3xl font-extrabold text-white">
                              {(simResult.risk_index * 100).toFixed(1)}%
                            </div>
                            <span className="text-[11px] text-gray-500">{simResult.execution_time_ms} ms inference</span>
                          </div>
                        </div>

                        <div className="p-4 rounded-xl border border-gray-800 bg-[#0B0F17]/70 space-y-2">
                          <span className="text-xs font-bold text-gray-300 flex items-center gap-1.5">
                            <Info className="w-4 h-4 text-emerald-400" />
                            Compliance Narrative (Audit Statement)
                          </span>
                          <p className="text-sm text-gray-300 leading-relaxed">
                            {simResult.shap_explanation?.human_readable_narrative}
                          </p>
                        </div>

                        <div>
                          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Multi-Model Ensemble Votes</span>
                          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-2">
                            {Object.entries(simResult.ensemble_breakdown || {}).map(([model, score]) => (
                              <div key={model} className="p-3 rounded-xl bg-gray-800/50 border border-gray-700/50 text-center">
                                <span className="text-[11px] text-gray-400 uppercase block truncate">{model}</span>
                                <span className={`text-base font-bold mt-1 block ${score > 0.5 ? 'text-rose-400' : 'text-emerald-400'}`}>
                                  {(score * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {simResult.shap_explanation?.top_features && (
                          <div>
                            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Top TreeSHAP Marginal Attributions</span>
                            <div className="space-y-1.5 mt-2">
                              {simResult.shap_explanation.top_features.map((feat, i) => (
                                <div key={i} className="flex items-center justify-between text-xs p-2 rounded-lg bg-gray-800/30 border border-gray-800">
                                  <span className="font-mono text-gray-300">{feat.feature}</span>
                                  <div className="flex items-center gap-2">
                                    <span className="text-gray-400 font-mono">val: {feat.value}</span>
                                    <span className={`font-mono font-bold ${feat.contribution > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                                      {feat.contribution > 0 ? `+${feat.contribution.toFixed(4)}` : feat.contribution.toFixed(4)}
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {simResult.rule_triggers && simResult.rule_triggers.length > 0 && (
                          <div>
                            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Heuristic Rule Triggers</span>
                            <div className="flex flex-wrap gap-1.5 mt-2">
                              {simResult.rule_triggers.map((rule, idx) => (
                                <span key={idx} className="text-xs px-2.5 py-1 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                                  {rule}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-2xl p-12 border border-gray-800 bg-[#111827]/40 flex flex-col items-center justify-center text-center space-y-3">
                      <Zap className="w-12 h-12 text-gray-600" />
                      <h5 className="text-lg font-semibold text-gray-300">Awaiting Simulation Trigger</h5>
                      <p className="text-sm text-gray-500 max-w-sm">
                        Select a transaction preset or enter custom attributes, then click "Run Multi-Model Fraud Audit" to evaluate risk.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {tab === 'ledger' && (
              <div className="rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-4">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-2 w-full sm:w-auto">
                    <div className="relative flex-1 sm:w-64">
                      <Search className="w-4 h-4 text-gray-500 absolute left-3 top-3" />
                      <input
                        type="text"
                        placeholder="Search origin, dest, type..."
                        value={ledgerSearch}
                        onChange={e => setLedgerSearch(e.target.value)}
                        className="w-full pl-9 pr-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-xs text-white focus:outline-none focus:border-emerald-500"
                      />
                    </div>
                    <select
                      value={ledgerFilter}
                      onChange={e => setLedgerFilter(e.target.value)}
                      className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
                    >
                      <option value="ALL">All Risk Tiers</option>
                      <option value="CRITICAL">Critical</option>
                      <option value="HIGH">High</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="LOW">Low</option>
                    </select>
                  </div>

                  <button
                    onClick={fetchTransactions}
                    className="text-xs px-3 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 flex items-center gap-1.5"
                  >
                    <RefreshCw className="w-3.5 h-3.5" /> Reload Ledger
                  </button>
                </div>

                {ledgerLoading ? (
                  <div className="p-8 text-center text-gray-400">Loading ledger records...</div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-gray-800 text-gray-400 font-semibold uppercase">
                          <th className="py-3 px-3">Tx ID</th>
                          <th className="py-3 px-3">Type</th>
                          <th className="py-3 px-3">Amount</th>
                          <th className="py-3 px-3">Origin</th>
                          <th className="py-3 px-3">Destination</th>
                          <th className="py-3 px-3">Risk Index</th>
                          <th className="py-3 px-3">Tier</th>
                          <th className="py-3 px-3">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/60 font-mono">
                        {filteredTransactions.map(tx => (
                          <tr key={tx.id} className="hover:bg-gray-800/30">
                            <td className="py-3 px-3 text-gray-300">#{tx.id}</td>
                            <td className="py-3 px-3 font-sans font-semibold text-gray-200">{tx.type}</td>
                            <td className="py-3 px-3 text-white font-bold">${tx.amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                            <td className="py-3 px-3 text-gray-400">{tx.nameOrig}</td>
                            <td className="py-3 px-3 text-gray-400">{tx.nameDest}</td>
                            <td className="py-3 px-3">
                              <span className={`font-bold ${(tx.risk_index || 0) > 0.5 ? 'text-rose-400' : 'text-emerald-400'}`}>
                                {((tx.risk_index || 0) * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="py-3 px-3 font-sans">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getTierColor(tx.risk_tier)}`}>
                                {tx.risk_tier}
                              </span>
                            </td>
                            <td className="py-3 px-3 font-sans">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${getRecColor(tx.recommendation)}`}>
                                {tx.recommendation}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {tab === 'network' && (
              <div className="rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-bold text-white flex items-center gap-2">
                      <Share2 className="w-5 h-5 text-rose-400" />
                      Relational Mule Network Ring (PyTorch GNN Topology)
                    </h4>
                    <p className="text-xs text-gray-400 mt-1">
                      Visualizing bipartite transaction graph: red nodes indicate high-risk mule hub targets.
                    </p>
                  </div>
                  <button onClick={fetchGraph} className="text-xs text-gray-400 hover:text-white flex items-center gap-1">
                    <RefreshCw className="w-3.5 h-3.5" /> Recompute Topology
                  </button>
                </div>

                <div className="rounded-xl border border-gray-800 bg-[#0B0F17] p-2 flex justify-center items-center overflow-hidden">
                  <canvas ref={canvasRef} width={800} height={450} className="w-full max-w-4xl h-auto" />
                </div>
              </div>
            )}

            {tab === 'models' && (
              <div className="rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-bold text-white flex items-center gap-2">
                      <Cpu className="w-5 h-5 text-purple-400" />
                      Model Benchmark Leaderboard (Held-Out Test Set: 6,000 Transactions)
                    </h4>
                    <p className="text-xs text-gray-400 mt-1">
                      Non-fabricated evaluation metrics trained strictly on zero-leakage financial split.
                    </p>
                  </div>
                  <button onClick={fetchMetrics} className="text-xs text-gray-400 hover:text-white flex items-center gap-1">
                    <RefreshCw className="w-3.5 h-3.5" /> Refresh
                  </button>
                </div>

                {metricsLoading ? (
                  <div className="p-8 text-center text-gray-400">Loading benchmark metrics...</div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-gray-800 text-gray-400 font-semibold uppercase">
                          <th className="py-3 px-3">Model</th>
                          <th className="py-3 px-3">Family</th>
                          <th className="py-3 px-3">Precision</th>
                          <th className="py-3 px-3">Recall</th>
                          <th className="py-3 px-3">F1-Score</th>
                          <th className="py-3 px-3">PR-AUC</th>
                          <th className="py-3 px-3">ROC-AUC</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/60 font-mono">
                        {metrics?.models && Object.entries(metrics.models).map(([name, m]) => (
                          <tr key={name} className="hover:bg-gray-800/30">
                            <td className="py-3 px-3 font-sans font-bold text-white">{name}</td>
                            <td className="py-3 px-3 font-sans text-gray-400">{m.family || 'Machine Learning'}</td>
                            <td className="py-3 px-3 text-emerald-400 font-bold">{m.precision?.toFixed(4)}</td>
                            <td className="py-3 px-3 text-emerald-400 font-bold">{m.recall?.toFixed(4)}</td>
                            <td className="py-3 px-3 text-teal-300 font-bold">{m.f1?.toFixed(4)}</td>
                            <td className="py-3 px-3 text-purple-300 font-bold">{m.pr_auc?.toFixed(4)}</td>
                            <td className="py-3 px-3 text-indigo-300 font-bold">{m.roc_auc?.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {tab === 'batch' && (
              <div className="rounded-2xl p-6 border border-gray-800 bg-[#111827]/70 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-bold text-white flex items-center gap-2">
                      <FileSpreadsheet className="w-5 h-5 text-teal-400" />
                      High-Throughput Batch Processing Studio
                    </h4>
                    <p className="text-xs text-gray-400 mt-1">
                      Execute high-concurrency fraud scoring across incoming transaction batches.
                    </p>
                  </div>
                  <button
                    onClick={handleRunBatchSample}
                    disabled={batchProcessing}
                    className="text-xs px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold flex items-center gap-1.5 transition disabled:opacity-50"
                  >
                    {batchProcessing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    Run 5-Transaction Sample Batch
                  </button>
                </div>

                {batchResults && (
                  <div className="space-y-4 pt-2">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="p-4 rounded-xl bg-gray-800/40 border border-gray-700">
                        <span className="text-xs text-gray-400">Total In Batch</span>
                        <div className="text-xl font-bold text-white mt-1">{batchResults.total_transactions}</div>
                      </div>
                      <div className="p-4 rounded-xl bg-gray-800/40 border border-gray-700">
                        <span className="text-xs text-gray-400">Total Latency</span>
                        <div className="text-xl font-bold text-teal-300 mt-1">{batchResults.total_execution_time_ms} ms</div>
                      </div>
                      <div className="p-4 rounded-xl bg-gray-800/40 border border-gray-700">
                        <span className="text-xs text-gray-400">Fraud Flagged</span>
                        <div className="text-xl font-bold text-rose-400 mt-1">{batchResults.fraud_count}</div>
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="border-b border-gray-800 text-gray-400 font-semibold uppercase">
                            <th className="py-2 px-3">Tx ID</th>
                            <th className="py-2 px-3">Risk Index</th>
                            <th className="py-2 px-3">Tier</th>
                            <th className="py-2 px-3">Decision</th>
                            <th className="py-2 px-3">Rules Fired</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-800/60 font-mono">
                          {batchResults.results?.map((res, i) => (
                            <tr key={i} className="hover:bg-gray-800/30">
                              <td className="py-2.5 px-3 text-gray-300">#{res.transaction_id}</td>
                              <td className="py-2.5 px-3">
                                <span className={`font-bold ${(res.risk_index || 0) > 0.5 ? 'text-rose-400' : 'text-emerald-400'}`}>
                                  {((res.risk_index || 0) * 100).toFixed(1)}%
                                </span>
                              </td>
                              <td className="py-2.5 px-3 font-sans">
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getTierColor(res.risk_tier)}`}>
                                  {res.risk_tier}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 font-sans">
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${getRecColor(res.recommendation)}`}>
                                  {res.recommendation}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-gray-400 font-sans">
                                {res.rule_triggers && res.rule_triggers.length > 0 ? res.rule_triggers.join(', ') : 'None'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
