// ─── Polypharmacy Safety Agent — React Frontend ───────────────────────────────
// Single-file app: React 18 + Tailwind CSS + native WebSocket + fetch()
// Backend: FastAPI at http://localhost:8000

const { useState, useEffect, useRef, useCallback, useReducer } = React;

const API = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// ─── Colour / badge helpers ────────────────────────────────────────────────────

const SEV_CONFIG = {
  CRITICAL: { bg: 'bg-crit-50', border: 'border-crit-300', badge: 'bg-crit-500 text-white', dot: 'bg-crit-500', icon: '🚨', label: 'CRITICAL' },
  MODERATE: { bg: 'bg-amber-50',  border: 'border-amber-200', badge: 'bg-amber-500 text-white',  dot: 'bg-amber-500',  icon: '⚠️', label: 'MODERATE' },
  NONE:     { bg: 'bg-teal-50',   border: 'border-teal-200',  badge: 'bg-teal-500 text-white',   dot: 'bg-teal-500',   icon: '✅', label: 'SAFE' },
};

function SeverityBadge({ severity, size = 'sm' }) {
  const cfg = SEV_CONFIG[severity] || SEV_CONFIG.NONE;
  const cls = size === 'lg' ? 'px-3 py-1 text-sm font-semibold' : 'px-2 py-0.5 text-xs font-semibold';
  return <span className={`${cfg.badge} ${cls} rounded-full inline-flex items-center gap-1`}>{cfg.icon} {cfg.label}</span>;
}

// ─── Reusable UI atoms ─────────────────────────────────────────────────────────

function Card({ children, className = '' }) {
  return <div className={`bg-white rounded-xl shadow-sm border border-slate-100 ${className}`}>{children}</div>;
}

function Spinner({ size = 5 }) {
  return (
    <svg className={`animate-spin w-${size} h-${size} text-teal-500`} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
    </svg>
  );
}

function Button({ children, onClick, disabled, variant = 'primary', size = 'md', className = '' }) {
  const base = 'rounded-lg font-medium transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2';
  const sizes = { sm: 'px-3 py-1.5 text-sm', md: 'px-4 py-2 text-sm', lg: 'px-6 py-3 text-base' };
  const variants = {
    primary:   'bg-teal-500 hover:bg-teal-600 text-white shadow-sm hover:shadow',
    secondary: 'bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 shadow-sm',
    danger:    'bg-crit-500 hover:bg-crit-600 text-white shadow-sm',
    ghost:     'text-teal-600 hover:bg-teal-50',
  };
  return (
    <button className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

function Input({ label, value, onChange, placeholder, required, type = 'text', hint }) {
  return (
    <div>
      {label && <label className="block text-sm font-medium text-slate-700 mb-1">{label}{required && <span className="text-crit-500 ml-0.5">*</span>}</label>}
      <input
        type={type} value={value} onChange={e => onChange(e.target.value)}
        placeholder={placeholder} required={required}
        className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400 focus:border-transparent transition"
      />
      {hint && <p className="text-xs text-slate-400 mt-1">{hint}</p>}
    </div>
  );
}

function Tabs({ tabs, active, onChange }) {
  return (
    <div className="flex border-b border-slate-200 gap-1">
      {tabs.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-all -mb-px ${active === t.id ? 'border-teal-500 text-teal-600' : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'}`}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ─── Sidebar navigation ────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { id: 'dashboard',   label: 'Dashboard',       icon: '▦' },
  { id: 'patient',     label: 'Patient Portal',  icon: '⊕' },
  { id: 'doctor',      label: 'Doctor Station',  icon: '⚕' },
  { id: 'healthcard',  label: 'Health Card',     icon: '♥' },
  { id: 'audit',       label: 'Audit Trail',     icon: '☰' },
];

function Sidebar({ view, onChange }) {
  return (
    <aside className="w-56 bg-teal-900 text-white flex flex-col min-h-screen fixed left-0 top-0 z-30">
      <div className="px-5 py-5 border-b border-teal-700">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-teal-400 rounded-lg flex items-center justify-center text-teal-900 font-bold text-sm">Rx</div>
          <div>
            <div className="font-semibold text-sm leading-tight">Polypharmacy</div>
            <div className="text-teal-300 text-xs">Safety Agent</div>
          </div>
        </div>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV_ITEMS.map(item => (
          <button key={item.id} onClick={() => onChange(item.id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all
              ${view === item.id ? 'bg-teal-700 text-white' : 'text-teal-200 hover:bg-teal-800 hover:text-white'}`}>
            <span className="text-base w-5 text-center">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>
      <div className="px-5 py-4 border-t border-teal-700">
        <div className="text-xs text-teal-400">API: {API}</div>
      </div>
    </aside>
  );
}

// ─── DASHBOARD ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color = 'teal', icon }) {
  const colors = {
    teal:  'from-teal-500 to-teal-600',
    red:   'from-crit-500 to-crit-600',
    green: 'from-emerald-500 to-emerald-600',
    amber: 'from-amber-400 to-amber-500',
  };
  return (
    <Card className="overflow-hidden">
      <div className={`bg-gradient-to-r ${colors[color]} p-4 text-white`}>
        <div className="flex justify-between items-start">
          <div>
            <p className="text-sm opacity-80 font-medium">{label}</p>
            <p className="text-3xl font-bold mt-1">{value}</p>
            {sub && <p className="text-xs opacity-70 mt-1">{sub}</p>}
          </div>
          <span className="text-2xl opacity-70">{icon}</span>
        </div>
      </div>
    </Card>
  );
}

function Dashboard({ scanHistory }) {
  const total    = scanHistory.length;
  const critical = scanHistory.filter(s => s.overall_severity === 'CRITICAL').length;
  const safe     = scanHistory.filter(s => s.overall_severity === 'NONE').length;
  const moderate = scanHistory.filter(s => s.overall_severity === 'MODERATE').length;

  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-xl font-semibold text-slate-800">Dashboard</h1>
        <p className="text-sm text-slate-500 mt-0.5">Medication safety overview</p>
      </div>

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard label="Total Scans"      value={total}    sub="all time"          icon="⊞" color="teal"  />
        <StatCard label="Critical Alerts"  value={critical} sub="require action"    icon="🚨" color="red"   />
        <StatCard label="Safe Checks"      value={safe}     sub="no interactions"   icon="✅" color="green" />
        <StatCard label="Moderate Alerts"  value={moderate} sub="monitoring needed" icon="⚠️" color="amber" />
      </div>

      <Card>
        <div className="px-5 py-4 border-b border-slate-100">
          <h2 className="font-semibold text-slate-700">Recent Scans</h2>
        </div>
        {scanHistory.length === 0 ? (
          <div className="py-16 text-center text-slate-400">
            <div className="text-4xl mb-3">📋</div>
            <p className="font-medium">No scans yet</p>
            <p className="text-sm mt-1">Run a scan from Patient Portal to see results here</p>
          </div>
        ) : (
          <div className="divide-y divide-slate-50">
            {[...scanHistory].reverse().map((scan, i) => (
              <div key={i} className="px-5 py-3.5 flex items-center justify-between hover:bg-slate-50 transition">
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${SEV_CONFIG[scan.overall_severity]?.dot || 'bg-slate-300'}`} />
                  <div>
                    <span className="text-sm font-medium text-slate-800">{scan.patient_id}</span>
                    <span className="text-xs text-slate-400 ml-2">{scan.processing_time_ms}ms</span>
                  </div>
                  <span className="text-xs text-slate-400">{scan.conflicts?.length || 0} conflict{scan.conflicts?.length !== 1 ? 's' : ''}</span>
                </div>
                <div className="flex items-center gap-3">
                  <SeverityBadge severity={scan.overall_severity} />
                  <span className="text-xs text-slate-400">{new Date(scan.scanned_at).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

// ─── AGENT TRACE PANEL ─────────────────────────────────────────────────────────

const AGENT_STEPS = [
  { id: 'profile_builder',     label: 'Profile Builder',     desc: 'Loading medications & normalising brand names' },
  { id: 'interaction_auditor', label: 'Interaction Auditor', desc: 'Running rule engine + Claude semantic check' },
  { id: 'report_generator',    label: 'Report Generator',    desc: 'Generating tiered reports for all stakeholders' },
];

const STEP_STATE = { waiting: 'waiting', running: 'running', complete: 'complete', error: 'error' };

function AgentStepCard({ step, state, message, index }) {
  const isRunning  = state === STEP_STATE.running;
  const isDone     = state === STEP_STATE.complete;
  const isError    = state === STEP_STATE.error;
  const isWaiting  = state === STEP_STATE.waiting;

  const iconMap = { waiting: '○', running: null, complete: '✓', error: '✕' };
  const ringColors = { waiting: 'border-slate-200', running: 'border-teal-400', complete: 'border-teal-500', error: 'border-crit-400' };
  const bgColors   = { waiting: 'bg-slate-50', running: 'bg-teal-50', complete: 'bg-teal-50', error: 'bg-crit-50' };
  const textColors = { waiting: 'text-slate-400', running: 'text-teal-600', complete: 'text-teal-700', error: 'text-crit-600' };

  return (
    <div className={`flex items-start gap-3 p-3.5 rounded-lg border ${ringColors[state]} ${bgColors[state]} transition-all duration-300 ${isDone || isRunning ? 'slide-in' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 border-2 ${ringColors[state]} ${bgColors[state]}`}>
        {isRunning ? <Spinner size={4} /> : <span className={`text-sm font-bold ${textColors[state]}`}>{iconMap[state]}</span>}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-semibold ${textColors[state]}`}>{step.label}</span>
          {isDone && <span className="text-xs text-teal-500 font-medium">Done</span>}
          {isRunning && <span className="text-xs text-teal-400 pulse-ring font-medium">Running…</span>}
        </div>
        <p className="text-xs text-slate-500 mt-0.5">{message || step.desc}</p>
      </div>
      <span className="text-xs text-slate-300 font-mono">#{index + 1}</span>
    </div>
  );
}

// ─── DROP ZONE ─────────────────────────────────────────────────────────────────

function DropZone({ onJson }) {
  const [active, setActive] = useState(false);
  const [filename, setFilename]  = useState('');
  const [pasted, setPasted] = useState('');
  const [error, setError] = useState('');
  const inputRef = useRef();

  const processFile = file => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const parsed = JSON.parse(e.target.result);
        setFilename(file.name);
        setError('');
        onJson(parsed);
      } catch {
        setError('Invalid JSON file.');
      }
    };
    reader.readAsText(file);
  };

  const handleDrop = e => {
    e.preventDefault(); setActive(false);
    processFile(e.dataTransfer.files[0]);
  };

  const handlePaste = val => {
    setPasted(val);
    if (!val.trim()) return;
    try {
      const parsed = JSON.parse(val);
      setError('');
      onJson(parsed);
    } catch {
      setError('Invalid JSON.');
    }
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={e => { e.preventDefault(); setActive(true); }}
        onDragLeave={() => setActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${active ? 'drop-zone-active' : 'border-slate-200 hover:border-teal-300 hover:bg-teal-50'}`}>
        <input ref={inputRef} type="file" accept=".json" className="hidden" onChange={e => processFile(e.target.files[0])} />
        <div className="text-2xl mb-2">📄</div>
        {filename
          ? <p className="text-sm font-medium text-teal-600">{filename}</p>
          : <><p className="text-sm font-medium text-slate-600">Drop JSON file or click to browse</p>
             <p className="text-xs text-slate-400 mt-1">FHIR-lite patient format</p></>
        }
      </div>
      <div>
        <p className="text-xs text-slate-500 mb-1">Or paste JSON directly:</p>
        <textarea
          rows={4} value={pasted} onChange={e => handlePaste(e.target.value)}
          placeholder='{"id":"patient-001","medications":[...]}'
          className="w-full border border-slate-200 rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-teal-400 resize-none" />
      </div>
      {error && <p className="text-xs text-crit-500">{error}</p>}
    </div>
  );
}

// ─── CONFLICT LIST ─────────────────────────────────────────────────────────────

function ConflictCard({ conflict, index }) {
  const [open, setOpen] = useState(false);
  const sev = conflict.severity || 'MODERATE';
  const cfg = SEV_CONFIG[sev] || SEV_CONFIG.MODERATE;

  return (
    <div className={`rounded-lg border ${cfg.border} ${cfg.bg} overflow-hidden`}>
      <button onClick={() => setOpen(o => !o)} className="w-full text-left px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-lg">{cfg.icon}</span>
          <div>
            <span className="text-sm font-semibold text-slate-800">{conflict.drug_a} + {conflict.drug_b}</span>
            <span className="ml-2 text-xs text-slate-500">{conflict.conflict_type === 'ALLERGY' ? '• Allergy' : `• ${conflict.rule_id || 'Semantic'}`}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <SeverityBadge severity={sev} />
          <span className="text-slate-400 text-xs">{open ? '▲' : '▼'}</span>
        </div>
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-2 border-t border-slate-100 pt-3">
          {conflict.mechanism && (
            <div>
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-1">Mechanism</p>
              <p className="text-sm text-slate-700">{conflict.mechanism}</p>
            </div>
          )}
          {conflict.clinical_effects?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-1">Clinical Effects</p>
              <div className="flex flex-wrap gap-1.5">
                {conflict.clinical_effects.map((e, i) => (
                  <span key={i} className="text-xs bg-white border border-slate-200 rounded-full px-2.5 py-0.5 text-slate-600">{e}</span>
                ))}
              </div>
            </div>
          )}
          {conflict.suggested_alternative && (
            <div className="bg-white rounded-lg p-3 border border-teal-100">
              <p className="text-xs font-semibold text-teal-700 uppercase tracking-wide mb-1">Suggested Alternative</p>
              <p className="text-xs text-slate-700">{conflict.suggested_alternative}</p>
            </div>
          )}
          {conflict.monitoring && (
            <div>
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-1">Monitoring</p>
              <p className="text-xs text-slate-600">{conflict.monitoring}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── REPORT TABS ───────────────────────────────────────────────────────────────

function ReportTabs({ reports }) {
  const [activeTab, setActiveTab] = useState('patient');
  const TABS = [
    { id: 'patient',     label: '👤 Patient' },
    { id: 'coordinator', label: '🏥 Coordinator' },
    { id: 'physician',   label: '⚕ Physician' },
  ];
  return (
    <div>
      <Tabs tabs={TABS} active={activeTab} onChange={setActiveTab} />
      <div className="pt-4">
        <div className="bg-slate-50 rounded-xl p-4 text-sm text-slate-700 leading-relaxed whitespace-pre-wrap border border-slate-100 max-h-72 overflow-y-auto">
          {reports[activeTab] || <span className="text-slate-400 italic">No report available.</span>}
        </div>
      </div>
    </div>
  );
}

// ─── PATIENT PORTAL ────────────────────────────────────────────────────────────

function PatientPortal({ onScanComplete }) {
  const [patientId, setPatientId] = useState('');
  const [patientJson, setPatientJson] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState('');
  const [result, setResult] = useState(null);
  const [agentSteps, setAgentSteps] = useState(
    AGENT_STEPS.reduce((acc, s) => ({ ...acc, [s.id]: { state: STEP_STATE.waiting, message: '' } }), {})
  );
  const wsRef = useRef(null);

  const connectWs = id => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`${WS_BASE}/api/ws/${id}`);
    ws.onmessage = e => {
      try {
        const data = JSON.parse(e.data);
        if (data.event === 'agent_complete') {
          setAgentSteps(prev => ({
            ...prev,
            [data.agent]: { state: STEP_STATE.complete, message: data.message },
          }));
          // Mark next step as running
          const idx = AGENT_STEPS.findIndex(s => s.id === data.agent);
          if (idx >= 0 && idx < AGENT_STEPS.length - 1) {
            const next = AGENT_STEPS[idx + 1].id;
            setAgentSteps(prev => ({ ...prev, [next]: { state: STEP_STATE.running, message: '' } }));
          }
        }
      } catch {}
    };
    wsRef.current = ws;
  };

  const resetSteps = () => {
    const fresh = AGENT_STEPS.reduce((acc, s) => ({ ...acc, [s.id]: { state: STEP_STATE.waiting, message: '' } }), {});
    setAgentSteps({ ...fresh, profile_builder: { state: STEP_STATE.running, message: '' } });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    if (!patientId.trim()) { setError('Patient ID is required.'); return; }
    if (!patientJson)      { setError('Please upload or paste a patient JSON file.'); return; }
    setError(''); setResult(null);
    setLoading(true);
    resetSteps();
    connectWs(patientId.trim());

    const body = {
      patient_id: patientId.trim(),
      prescriptions: patientJson.medications || [],
      patient_name: patientJson.name?.[0]
        ? `${patientJson.name[0].given?.[0] || ''} ${patientJson.name[0].family || ''}`.trim()
        : undefined,
    };

    try {
      const res = await fetch(`${API}/api/patient/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult({ ...data, scanned_at: new Date().toISOString() });
      onScanComplete({ ...data, scanned_at: new Date().toISOString() });
      // Mark all steps complete
      setAgentSteps(AGENT_STEPS.reduce((acc, s) => ({ ...acc, [s.id]: { state: STEP_STATE.complete, message: '' } }), {}));
    } catch (err) {
      setError(err.message);
      setAgentSteps(prev => {
        const running = Object.entries(prev).find(([,v]) => v.state === STEP_STATE.running);
        if (!running) return prev;
        return { ...prev, [running[0]]: { state: STEP_STATE.error, message: err.message } };
      });
    } finally {
      setLoading(false);
      if (wsRef.current) wsRef.current.close();
    }
  };

  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-xl font-semibold text-slate-800">Patient Portal</h1>
        <p className="text-sm text-slate-500 mt-0.5">Upload a patient record to run a full polypharmacy safety scan</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Left — form */}
        <Card className="p-5">
          <h2 className="font-semibold text-slate-700 mb-4">Scan Input</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Input label="Patient ID" value={patientId} onChange={setPatientId}
              placeholder="e.g. patient-001" required hint="Must match the id field in your JSON" />
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient JSON <span className="text-crit-500">*</span></label>
              <DropZone onJson={setPatientJson} />
            </div>
            {error && <div className="text-sm text-crit-600 bg-crit-50 border border-crit-200 rounded-lg px-3 py-2">{error}</div>}
            {patientJson && (
              <div className="text-xs text-teal-700 bg-teal-50 border border-teal-200 rounded-lg px-3 py-2">
                ✓ JSON loaded — {patientJson.medications?.length || 0} medication(s) found
              </div>
            )}
            <Button type="submit" disabled={loading} size="lg" className="w-full justify-center">
              {loading ? <><Spinner size={4} /> Running scan…</> : '▶ Run Safety Scan'}
            </Button>
          </form>
        </Card>

        {/* Right — agent trace */}
        <Card className="p-5">
          <h2 className="font-semibold text-slate-700 mb-4">Agent Trace</h2>
          <div className="space-y-2.5">
            {AGENT_STEPS.map((step, i) => (
              <AgentStepCard key={step.id} step={step} index={i}
                state={agentSteps[step.id]?.state || STEP_STATE.waiting}
                message={agentSteps[step.id]?.message} />
            ))}
          </div>
          {!loading && !result && (
            <p className="text-xs text-slate-400 text-center mt-4">Submit a scan to see live agent progress</p>
          )}
        </Card>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-5 fade-in">
          {/* Summary bar */}
          <Card className="p-5">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <h2 className="font-semibold text-slate-700">Scan Results — {result.patient_id}</h2>
                <p className="text-xs text-slate-400 mt-0.5">Processed in {result.processing_time_ms}ms · {result.conflicts?.length || 0} interaction(s) found</p>
              </div>
              <SeverityBadge severity={result.overall_severity} size="lg" />
            </div>
          </Card>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* Conflicts */}
            <Card className="p-5">
              <h3 className="font-semibold text-slate-700 mb-3">Detected Interactions</h3>
              {result.conflicts?.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <div className="text-3xl mb-2">✅</div>
                  <p className="text-sm font-medium">No interactions detected</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {result.conflicts.map((c, i) => <ConflictCard key={i} conflict={c} index={i} />)}
                </div>
              )}
            </Card>
            {/* Reports */}
            <Card className="p-5">
              <h3 className="font-semibold text-slate-700 mb-3">Reports</h3>
              <ReportTabs reports={result.reports || {}} />
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── DOCTOR STATION ────────────────────────────────────────────────────────────

function CriticalAlertBanner({ conflicts, onDismiss }) {
  const [ackText, setAckText] = useState('');
  const critical = conflicts.filter(c => c.severity === 'CRITICAL');

  return (
    <div className="bg-crit-50 border-2 border-crit-400 rounded-xl p-5 space-y-4 fade-in">
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 bg-crit-500 rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-white text-xl">!</span>
        </div>
        <div>
          <h3 className="text-lg font-bold text-crit-700">CRITICAL Drug Interaction Detected</h3>
          <p className="text-sm text-crit-600 mt-0.5">This prescription must NOT be dispensed until the interaction is reviewed.</p>
        </div>
      </div>
      <div className="space-y-3">
        {critical.map((c, i) => (
          <div key={i} className="bg-white border border-crit-200 rounded-lg p-3">
            <div className="font-semibold text-crit-700 text-sm">{c.drug_a} + {c.drug_b}</div>
            <p className="text-xs text-slate-600 mt-1">{c.mechanism}</p>
            {c.suggested_alternative && (
              <p className="text-xs text-teal-700 mt-1 font-medium">Alternative: {c.suggested_alternative}</p>
            )}
          </div>
        ))}
      </div>
      <div className="border-t border-crit-200 pt-4">
        <p className="text-sm text-crit-600 mb-2 font-medium">Type <strong>ACKNOWLEDGE</strong> to dismiss this alert and log your review:</p>
        <div className="flex gap-2">
          <input value={ackText} onChange={e => setAckText(e.target.value)}
            placeholder="Type ACKNOWLEDGE"
            className="flex-1 border border-crit-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-crit-400" />
          <Button variant="danger" disabled={ackText !== 'ACKNOWLEDGE'} onClick={onDismiss}>Acknowledge</Button>
        </div>
      </div>
    </div>
  );
}

function SafeConfirmBanner({ drugName, patientId }) {
  return (
    <div className="bg-teal-50 border-2 border-teal-300 rounded-xl p-5 fade-in">
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 bg-teal-500 rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-white text-xl">✓</span>
        </div>
        <div>
          <h3 className="text-lg font-bold text-teal-700">Safe to Prescribe</h3>
          <p className="text-sm text-teal-600 mt-1">
            <strong>{drugName}</strong> has been checked against the current medication record for <strong>{patientId}</strong>.
            No critical interactions detected. The prescription has been added to the patient record.
          </p>
        </div>
      </div>
    </div>
  );
}

function DoctorStation() {
  const [form, setForm] = useState({ patient_id: '', drug_name: '', dose: '', frequency: 'once daily', prescribing_doctor: '', condition: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const [result, setResult]   = useState(null);
  const [dismissed, setDismissed] = useState(false);

  const set = k => v => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = async e => {
    e.preventDefault();
    const missing = Object.entries(form).filter(([k,v]) => k !== 'frequency' && !v.trim()).map(([k]) => k);
    if (missing.length) { setError(`Required fields: ${missing.join(', ')}`); return; }
    setError(''); setResult(null); setDismissed(false);
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/doctor/check`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, prescription_date: new Date().toISOString().slice(0, 10) }),
      });
      if (res.status === 404) throw new Error('Patient profile not found. Run a Patient Scan first.');
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-xl font-semibold text-slate-800">Doctor Station</h1>
        <p className="text-sm text-slate-500 mt-0.5">Check a new prescription against a patient's existing medications</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card className="p-5">
          <h2 className="font-semibold text-slate-700 mb-4">New Prescription Check</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Input label="Patient ID" value={form.patient_id} onChange={set('patient_id')} placeholder="patient-001" required />
            <div className="grid grid-cols-2 gap-3">
              <Input label="Drug Name" value={form.drug_name} onChange={set('drug_name')} placeholder="Aspirin" required />
              <Input label="Dose" value={form.dose} onChange={set('dose')} placeholder="75mg" required />
            </div>
            <Input label="Frequency" value={form.frequency} onChange={set('frequency')} placeholder="once daily" />
            <Input label="Prescribing Doctor" value={form.prescribing_doctor} onChange={set('prescribing_doctor')} placeholder="Dr. Smith" required />
            <Input label="Condition" value={form.condition} onChange={set('condition')} placeholder="Hypertension" required />
            {error && <div className="text-sm text-crit-600 bg-crit-50 border border-crit-200 rounded-lg px-3 py-2">{error}</div>}
            <Button type="submit" disabled={loading} size="lg" className="w-full justify-center">
              {loading ? <><Spinner size={4} /> Checking…</> : '⚕ Check Prescription'}
            </Button>
          </form>
        </Card>

        <div className="space-y-4">
          {result && !dismissed && result.severity === 'CRITICAL' && (
            <CriticalAlertBanner conflicts={result.conflicts} onDismiss={() => setDismissed(true)} />
          )}
          {result && (dismissed || result.severity !== 'CRITICAL') && (
            <>
              {result.safe_to_prescribe
                ? <SafeConfirmBanner drugName={form.drug_name} patientId={form.patient_id} />
                : !dismissed && null
              }
              {result.conflicts?.length > 0 && (
                <Card className="p-5">
                  <h3 className="font-semibold text-slate-700 mb-3">Interactions Found</h3>
                  <div className="space-y-2">
                    {result.conflicts.map((c, i) => <ConflictCard key={i} conflict={c} />)}
                  </div>
                </Card>
              )}
              {result.physician_report && (
                <Card className="p-5">
                  <h3 className="font-semibold text-slate-700 mb-3">Physician Summary</h3>
                  <div className="bg-slate-50 rounded-xl p-4 text-sm text-slate-700 leading-relaxed whitespace-pre-wrap border border-slate-100 max-h-64 overflow-y-auto">
                    {result.physician_report}
                  </div>
                </Card>
              )}
            </>
          )}
          {!result && !loading && (
            <Card className="p-8 text-center text-slate-400">
              <div className="text-4xl mb-3">⚕</div>
              <p className="font-medium">Check result will appear here</p>
              <p className="text-sm mt-1">Submit a prescription to check for interactions</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── HEALTH CARD ───────────────────────────────────────────────────────────────

function HealthCard() {
  const [patientId, setPatientId] = useState('');
  const [loading, setLoading]     = useState(false);
  const [profile, setProfile]     = useState(null);
  const [conflicts, setConflicts] = useState([]);
  const [error, setError]         = useState('');

  const fetchProfile = async () => {
    if (!patientId.trim()) { setError('Enter a Patient ID.'); return; }
    setError(''); setLoading(true); setProfile(null); setConflicts([]);
    try {
      const [profRes, scanRes] = await Promise.allSettled([
        fetch(`${API}/api/patient/${patientId.trim()}/profile`),
        fetch(`${API}/api/patient/${patientId.trim()}/audit`),
      ]);
      if (profRes.status === 'fulfilled') {
        if (!profRes.value.ok) throw new Error(profRes.value.status === 404 ? 'Patient not found.' : `HTTP ${profRes.value.status}`);
        setProfile(await profRes.value.json());
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const checklist = profile?.medications?.filter(m => m.active_status) || [];

  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-xl font-semibold text-slate-800">Health Card</h1>
        <p className="text-sm text-slate-500 mt-0.5">Patient medication summary and safety checklist</p>
      </div>

      <Card className="p-5">
        <div className="flex gap-3">
          <div className="flex-1">
            <Input value={patientId} onChange={setPatientId} placeholder="Enter Patient ID…"
              hint="Fetches live data from Redis" />
          </div>
          <Button onClick={fetchProfile} disabled={loading} className="mt-0 self-end">
            {loading ? <Spinner size={4} /> : 'Load'}
          </Button>
        </div>
        {error && <p className="text-sm text-crit-600 mt-2">{error}</p>}
      </Card>

      {profile && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 fade-in">
          {/* Medication table */}
          <Card className="xl:col-span-2 overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100 flex justify-between items-center">
              <h2 className="font-semibold text-slate-700">Current Medications</h2>
              <span className="text-xs bg-teal-100 text-teal-700 rounded-full px-2.5 py-0.5 font-medium">
                {profile.medication_count} active
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50 text-xs text-slate-500 uppercase tracking-wide">
                    <th className="text-left px-5 py-3 font-medium">Drug</th>
                    <th className="text-left px-4 py-3 font-medium">Dose</th>
                    <th className="text-left px-4 py-3 font-medium hidden md:table-cell">Condition</th>
                    <th className="text-left px-4 py-3 font-medium hidden lg:table-cell">Prescribing Doctor</th>
                    <th className="text-left px-4 py-3 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {profile.medications.map((m, i) => (
                    <tr key={i} className="hover:bg-slate-50 transition">
                      <td className="px-5 py-3">
                        <div className="font-medium text-slate-800">{m.generic_name}</div>
                        {m.is_normalised && <div className="text-xs text-slate-400">was: {m.drug_name}</div>}
                      </td>
                      <td className="px-4 py-3 text-slate-600 font-mono text-xs">{m.dose}</td>
                      <td className="px-4 py-3 text-slate-500 hidden md:table-cell text-xs">{m.condition}</td>
                      <td className="px-4 py-3 text-slate-500 hidden lg:table-cell text-xs">{m.prescribing_doctor}</td>
                      <td className="px-4 py-3">
                        <span className={`text-xs rounded-full px-2 py-0.5 font-medium ${m.active_status ? 'bg-teal-100 text-teal-700' : 'bg-slate-100 text-slate-500'}`}>
                          {m.active_status ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Tell your doctor checklist */}
          <Card className="p-5">
            <h2 className="font-semibold text-slate-700 mb-1">Tell Your Doctor</h2>
            <p className="text-xs text-slate-400 mb-4">Bring this checklist to every appointment</p>
            <div className="space-y-2.5">
              {checklist.map((m, i) => (
                <label key={i} className="flex items-start gap-2.5 cursor-pointer group">
                  <input type="checkbox" className="mt-0.5 accent-teal-600 w-4 h-4 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-slate-700 group-hover:text-teal-600 transition">
                      {m.generic_name} {m.dose}
                    </p>
                    <p className="text-xs text-slate-400">{m.prescribing_doctor}</p>
                  </div>
                </label>
              ))}
            </div>
            <div className="mt-5 p-3 bg-teal-50 rounded-lg border border-teal-100">
              <p className="text-xs text-teal-700 font-medium">💡 Tip</p>
              <p className="text-xs text-teal-600 mt-1">Show this list to every doctor and pharmacist before starting any new medicine, vitamin, or supplement.</p>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

// ─── AUDIT TRAIL ───────────────────────────────────────────────────────────────

const NODE_ICONS = {
  profile_builder_node:      '👤',
  interaction_auditor_node:  '🔍',
  human_checkpoint_node:     '👁',
  report_generator_node:     '📝',
  immediate_alert_node:      '🚨',
  safe_confirm_node:         '✅',
  confirm_and_persist_node:  '💾',
};

const SEV_TIMELINE_COLOR = {
  CRITICAL: 'bg-crit-500',
  MODERATE: 'bg-amber-400',
  NONE:     'bg-teal-500',
  INFO:     'bg-slate-400',
};

function AuditTimeline() {
  const [patientId, setPatientId] = useState('');
  const [loading, setLoading]     = useState(false);
  const [entries, setEntries]     = useState(null);
  const [error, setError]         = useState('');

  const fetchAudit = async () => {
    if (!patientId.trim()) { setError('Enter a Patient ID.'); return; }
    setError(''); setLoading(true);
    try {
      const res = await fetch(`${API}/api/patient/${patientId.trim()}/audit`);
      if (res.status === 404) throw new Error('No audit trail found for this patient.');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setEntries(data.entries);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-xl font-semibold text-slate-800">Audit Trail</h1>
        <p className="text-sm text-slate-500 mt-0.5">Complete agent execution timeline for any patient</p>
      </div>

      <Card className="p-5">
        <div className="flex gap-3">
          <div className="flex-1">
            <Input value={patientId} onChange={setPatientId} placeholder="Enter Patient ID…" />
          </div>
          <Button onClick={fetchAudit} disabled={loading} className="self-end">
            {loading ? <Spinner size={4} /> : 'Load Trail'}
          </Button>
        </div>
        {error && <p className="text-sm text-crit-600 mt-2">{error}</p>}
      </Card>

      {entries && (
        <Card className="p-5 fade-in">
          <div className="flex justify-between items-center mb-5">
            <h2 className="font-semibold text-slate-700">Timeline — {patientId}</h2>
            <span className="text-xs bg-slate-100 text-slate-600 rounded-full px-2.5 py-1">{entries.length} events</span>
          </div>

          {entries.length === 0 ? (
            <p className="text-sm text-slate-400 text-center py-8">No audit events found.</p>
          ) : (
            <div className="relative">
              {/* Vertical line */}
              <div className="absolute left-5 top-0 bottom-0 w-px bg-slate-200" />
              <div className="space-y-4">
                {entries.map((entry, i) => {
                  const dotColor = SEV_TIMELINE_COLOR[entry.severity] || 'bg-slate-400';
                  const nodeIcon = NODE_ICONS[entry.node_name] || '⬡';
                  const snap = entry.state_snapshot;
                  return (
                    <div key={entry.id || i} className="flex gap-4 pl-2 slide-in">
                      <div className={`w-6 h-6 rounded-full ${dotColor} flex items-center justify-center flex-shrink-0 z-10 ring-2 ring-white`}>
                        <span className="text-white text-xs">{nodeIcon}</span>
                      </div>
                      <div className="flex-1 pb-4">
                        <div className="flex items-start justify-between gap-2 flex-wrap">
                          <div>
                            <span className="text-sm font-semibold text-slate-700">{entry.node_name}</span>
                            <span className="ml-2 text-xs text-slate-400 font-mono">
                              {new Date(entry.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <span className={`text-xs rounded-full px-2 py-0.5 font-medium ${
                            entry.severity === 'CRITICAL' ? 'bg-crit-100 text-crit-700' :
                            entry.severity === 'MODERATE' ? 'bg-amber-100 text-amber-700' :
                            entry.severity === 'NONE'     ? 'bg-teal-100 text-teal-700' :
                            'bg-slate-100 text-slate-600'}`}>
                            {entry.severity}
                          </span>
                        </div>
                        <p className="text-xs text-slate-600 mt-1">{entry.action}</p>
                        {snap && typeof snap === 'object' && Object.keys(snap).length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {Object.entries(snap).map(([k, v]) => (
                              <span key={k} className="text-xs bg-slate-100 text-slate-500 rounded px-2 py-0.5 font-mono">
                                {k}: {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

// ─── ROOT APP ──────────────────────────────────────────────────────────────────

function App() {
  const [view, setView]             = useState('dashboard');
  const [scanHistory, setScanHistory] = useState([]);

  const handleScanComplete = useCallback(result => {
    setScanHistory(h => [...h, result]);
  }, []);

  const VIEWS = {
    dashboard: <Dashboard scanHistory={scanHistory} />,
    patient:   <PatientPortal onScanComplete={handleScanComplete} />,
    doctor:    <DoctorStation />,
    healthcard: <HealthCard />,
    audit:     <AuditTimeline />,
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar view={view} onChange={setView} />
      <main className="ml-56 flex-1 p-6 max-w-screen-xl">
        {VIEWS[view]}
      </main>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
