import { useState } from 'react'
import PredictionForm from './components/PredictionForm'
import PredictionResult from './components/PredictionResult'
import type { PredictionOutput } from './types/passenger'
import './index.css'

function App() {
  const [result, setResult] = useState<PredictionOutput | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  return (
    <div style={{ 
      minHeight: '100vh',
      padding: '40px 20px',
      maxWidth: '800px',
      margin: '0 auto'
    }}>
      <header style={{ textAlign: 'center', marginBottom: '48px' }}>
        <div style={{ 
          color: 'var(--accent-gold)', 
          fontSize: '12px', 
          letterSpacing: '4px',
          marginBottom: '12px',
          textTransform: 'uppercase'
        }}>
          RMS Titanic — April 15, 1912
        </div>
        <h1 style={{ 
          fontSize: '48px', 
          fontWeight: 900,
          background: 'linear-gradient(135deg, #f0f4ff 0%, #7dd3fc 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '12px'
        }}>
          Survival Predictor
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '13px', letterSpacing: '1px' }}>
          Enter passenger details to predict survival probability
        </p>
      </header>

      <PredictionForm 
        onResult={setResult}
        onLoading={setIsLoading}
        onError={setError}
      />

      {error && (
        <div style={{
          marginTop: '24px',
          padding: '16px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          color: '#ef4444',
          fontSize: '13px'
        }}>
          {error}
        </div>
      )}

      {isLoading && (
        <div style={{ 
          textAlign: 'center', 
          marginTop: '32px',
          color: 'var(--accent-ice)',
          fontSize: '13px',
          letterSpacing: '2px'
        }}>
          CALCULATING FATE...
        </div>
      )}

      {result && !isLoading && (
        <PredictionResult result={result} />
      )}
    </div>
  )
}

export default App