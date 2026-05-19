import { useState } from 'react'

const API = 'http://localhost:8001'

// ── Dados de demonstração ──────────────────────────────────────────────────
const DEMO_TOPICS = [
  {
    name: 'Graph Theory',
    feedbacks: [
      { text: 'This module was brilliantly structured. The visual examples made graphs very intuitive.', rating: 5 },
      { text: 'Excellent explanations! I really enjoyed learning about graphs and their applications.', rating: 4 },
    ],
  },
  {
    name: 'Naive Bayes',
    feedbacks: [
      { text: 'Amazing content! The probabilistic reasoning was explained step by step very clearly.', rating: 5 },
      { text: 'Best module so far. The examples were practical and the pace was perfect.', rating: 5 },
    ],
  },
  {
    name: 'Fuzzy Logic',
    feedbacks: [
      { text: 'The material was okay. Some concepts were a bit abstract but generally acceptable.', rating: 3 },
      { text: 'Average module. It covered the basics but lacked depth in the examples.', rating: 3 },
    ],
  },
  {
    name: 'Search Algorithms',
    feedbacks: [
      { text: 'This was very confusing. The BFS and DFS examples were hard to follow.', rating: 2 },
      { text: 'Difficult to understand. The explanations were too fast and incomplete.', rating: 2 },
    ],
  },
  {
    name: 'Simulated Annealing',
    feedbacks: [
      { text: 'The module was satisfactory. Covered the topic adequately without excelling.', rating: 3 },
      { text: 'Decent content. Some useful information but the examples could be clearer.', rating: 3 },
    ],
  },
]

// ── Helpers de cor ─────────────────────────────────────────────────────────
const SENTIMENT_COLOR = { positive: '#4ade80', negative: '#f87171', neutral: '#94a3b8' }
const LEVEL_COLOR     = { Alto: '#4ade80', Médio: '#facc15', Baixo: '#f87171' }

function pct(v) { return `${(v * 100).toFixed(1)}%` }
function fix2(v) { return Number(v).toFixed(2) }

// ── Subcomponentes ─────────────────────────────────────────────────────────

function Badge({ label, color }) {
  return <span className="badge" style={{ background: color }}>{label}</span>
}

function Layer1Panel({ topics }) {
  return (
    <div className="panel">
      <div className="panel-header layer1-header">
        <span className="layer-tag">Camada I</span>
        <h3>NLP — Naive Bayes</h3>
      </div>
      <div className="panel-body">
        {topics.map((topic, ti) => (
          <div key={ti} className="topic-block">
            <p className="topic-block-name">{topic.name}</p>
            {topic.feedbacks.map((fb, fi) => (
              <div key={fi} className="fb-row">
                <Badge label={fb.layer1.sentiment} color={SENTIMENT_COLOR[fb.layer1.sentiment]} />
                <span className="fb-snippet">"{fb.text.slice(0, 55)}…"</span>
                <span className="proba-row">
                  <span style={{ color: '#4ade80' }}>pos {pct(fb.layer1.probabilities.positive ?? 0)}</span>
                  {' · '}
                  <span style={{ color: '#f87171' }}>neg {pct(fb.layer1.probabilities.negative ?? 0)}</span>
                  {' · '}
                  <span style={{ color: '#94a3b8' }}>neu {pct(fb.layer1.probabilities.neutral  ?? 0)}</span>
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

function Layer2Panel({ topics }) {
  return (
    <div className="panel">
      <div className="panel-header layer2-header">
        <span className="layer-tag">Camada II</span>
        <h3>Inferência Fuzzy — Mamdani</h3>
      </div>
      <div className="panel-body fuzzy-grid">
        {topics.map((topic, ti) => (
          <div key={ti} className="fuzzy-card">
            <p className="fuzzy-name">{topic.name}</p>
            <div className="fuzzy-score-row">
              <span className="fuzzy-score-value">{fix2(topic.layer2.engagement_score)}</span>
              <span className="fuzzy-score-max">/10</span>
            </div>
            <Badge label={topic.layer2.engagement_level} color={LEVEL_COLOR[topic.layer2.engagement_level]} />
            <div className="fuzzy-bar-wrap">
              <div className="fuzzy-bar" style={{ width: `${topic.layer2.engagement_score * 10}%` }} />
            </div>
            <p className="fuzzy-hint">
              sentimento médio: {pct(topic.avg_pos_prob)} positivo<br />
              nota média: {fix2(topic.avg_rating)} / 5
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

function Layer3Panel({ sa }) {
  const improvement = sa.initial_cost > 0
    ? (((sa.initial_cost - sa.final_cost) / sa.initial_cost) * 100).toFixed(1)
    : '0.0'

  return (
    <div className="panel">
      <div className="panel-header layer3-header">
        <span className="layer-tag">Camada III</span>
        <h3>Plano de Estudos — Simulated Annealing</h3>
      </div>
      <div className="panel-body">
        <div className="sa-stats">
          <span>Custo inicial: <strong>{fix2(sa.initial_cost)}</strong></span>
          <span className="sa-arrow">→</span>
          <span>Custo final: <strong>{fix2(sa.final_cost)}</strong></span>
          <span className="sa-improvement">↓ {improvement}% de melhora</span>
        </div>

        {sa.convergence.length > 0 && (
          <div className="conv-wrap">
            <p className="conv-label">Convergência do SA</p>
            <svg viewBox={`0 0 200 40`} className="conv-svg">
              {(() => {
                const pts = sa.convergence
                const max = Math.max(...pts)
                const min = Math.min(...pts)
                const range = max - min || 1
                const coords = pts.map((v, i) =>
                  `${(i / (pts.length - 1)) * 200},${40 - ((v - min) / range) * 36}`
                ).join(' ')
                return <polyline points={coords} fill="none" stroke="#6366f1" strokeWidth="1.5" />
              })()}
            </svg>
          </div>
        )}

        <ol className="study-plan">
          {sa.sequence.map((item, i) => (
            <li key={i} className="plan-item">
              <span className="plan-pos">{item.position}º</span>
              <span className="plan-name">{item.name}</span>
              <div className="plan-bar-wrap">
                <div className="plan-bar" style={{ width: `${item.engagement_score * 10}%` }} />
              </div>
              <span className="plan-score">{fix2(item.engagement_score)}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

// ── Componente principal ───────────────────────────────────────────────────

export default function App() {
  const [topicName,        setTopicName]        = useState('')
  const [feedbackText,     setFeedbackText]     = useState('')
  const [rating,           setRating]           = useState(3)
  const [pendingFeedbacks, setPendingFeedbacks] = useState([])
  const [topics,           setTopics]           = useState([])
  const [results,          setResults]          = useState(null)
  const [loading,          setLoading]          = useState(false)
  const [error,            setError]            = useState(null)

  function addFeedback() {
    if (!feedbackText.trim()) return
    setPendingFeedbacks(prev => [...prev, { text: feedbackText.trim(), rating: Number(rating) }])
    setFeedbackText('')
    setRating(3)
  }

  function addTopic() {
    if (!topicName.trim() || pendingFeedbacks.length === 0) return
    setTopics(prev => [...prev, { name: topicName.trim(), feedbacks: pendingFeedbacks }])
    setTopicName('')
    setPendingFeedbacks([])
  }

  function removeTopic(i) {
    setTopics(prev => prev.filter((_, idx) => idx !== i))
    setResults(null)
  }

  function loadDemo() {
    setTopics(DEMO_TOPICS)
    setResults(null)
    setError(null)
  }

  async function runPipeline() {
    if (topics.length < 2) {
      setError('Adicione ao menos 2 tópicos para gerar o plano de estudos.')
      return
    }
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const res = await fetch(`${API}/pipeline`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ topics }),
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(`Erro ${res.status}: ${msg}`)
      }
      setResults(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Sistema Inteligente de Triagem Educacional</h1>
        <p className="subtitle">NLP (Naive Bayes) → Lógica Fuzzy (Mamdani) → Simulated Annealing</p>
      </header>

      <main className="main">

        {/* ── Formulário de entrada ── */}
        <section className="card input-card">
          <h2>Adicionar Tópico</h2>
          <p className="hint">Os feedbacks devem estar em inglês (modelo treinado em inglês).</p>

          <label className="field-label">Nome do tópico</label>
          <input
            className="field-input"
            value={topicName}
            onChange={e => setTopicName(e.target.value)}
            placeholder="ex: Graph Theory"
            onKeyDown={e => e.key === 'Enter' && addFeedback()}
          />

          <label className="field-label">Feedback do aluno</label>
          <textarea
            className="field-textarea"
            rows={3}
            value={feedbackText}
            onChange={e => setFeedbackText(e.target.value)}
            placeholder="ex: This topic was very clear and well explained..."
          />

          <label className="field-label">
            Nota: <strong>{Number(rating).toFixed(1)}</strong> / 5
          </label>
          <input
            type="range" min={1} max={5} step={0.5}
            value={rating}
            onChange={e => setRating(e.target.value)}
            className="field-range"
          />

          {pendingFeedbacks.length > 0 && (
            <div className="pending-list">
              {pendingFeedbacks.map((fb, i) => (
                <div key={i} className="pending-item">
                  <span>⭐ {fb.rating} — "{fb.text.slice(0, 50)}…"</span>
                  <button className="btn-x" onClick={() => setPendingFeedbacks(prev => prev.filter((_, j) => j !== i))}>×</button>
                </div>
              ))}
            </div>
          )}

          <div className="btn-row">
            <button className="btn secondary" onClick={addFeedback} disabled={!feedbackText.trim()}>
              + Feedback
            </button>
            <button
              className="btn primary"
              onClick={addTopic}
              disabled={!topicName.trim() || pendingFeedbacks.length === 0}
            >
              + Tópico
            </button>
          </div>
        </section>

        {/* ── Lista de tópicos ── */}
        {topics.length > 0 && (
          <section className="card">
            <div className="topics-header">
              <h2>Tópicos adicionados ({topics.length})</h2>
              <button className="btn ghost" onClick={() => { setTopics([]); setResults(null) }}>Limpar todos</button>
            </div>
            <div className="topics-grid">
              {topics.map((t, i) => (
                <div key={i} className="topic-pill">
                  <span className="topic-pill-name">{t.name}</span>
                  <span className="topic-pill-count">{t.feedbacks.length} feedback(s)</span>
                  <button className="btn-x" onClick={() => removeTopic(i)}>×</button>
                </div>
              ))}
            </div>
            <div className="btn-row">
              <button className="btn ghost" onClick={loadDemo}>Carregar Demo</button>
              <button
                className="btn run"
                onClick={runPipeline}
                disabled={loading || topics.length < 2}
              >
                {loading ? '⏳ Processando…' : '▶ Executar Sistema'}
              </button>
            </div>
          </section>
        )}

        {topics.length === 0 && (
          <div className="empty-state">
            <p>Nenhum tópico adicionado ainda.</p>
            <button className="btn primary" onClick={loadDemo}>Carregar Demo (5 tópicos)</button>
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        {/* ── Resultados ── */}
        {results && (
          <section>
            <h2 className="results-title">Resultados do Pipeline</h2>
            <Layer1Panel topics={results.topics} />
            <Layer2Panel topics={results.topics} />
            <Layer3Panel sa={results.layer3} />
          </section>
        )}
      </main>
    </div>
  )
}
