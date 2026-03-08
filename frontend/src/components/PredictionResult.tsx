import type { PredictionOutput } from '../types/passenger'

interface Props {
  result: PredictionOutput
}

export default function PredictionResult({ result }: Props) {
  const percentage = Math.round(result.survival_probability * 100)
  const survived = result.survived

  return (
    <div style={{
      marginTop: '24px',
      background: 'var(--bg-card)',
      border: `1px solid ${survived ? 'rgba(125, 211, 252, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
      borderRadius: '12px',
      padding: '32px',
      textAlign: 'center',
    }}>

      <div style={{
        fontSize: '11px',
        letterSpacing: '4px',
        color: 'var(--text-muted)',
        textTransform: 'uppercase',
        marginBottom: '16px',
      }}>
        Prediction Result
      </div>

      <div style={{
        fontSize: '64px',
        fontFamily: 'Playfair Display, serif',
        fontWeight: 900,
        color: survived ? 'var(--accent-ice)' : 'var(--accent-danger)',
        marginBottom: '8px',
        lineHeight: 1,
      }}>
        {survived ? 'SURVIVED' : 'PERISHED'}
      </div>

      <div style={{
        fontSize: '13px',
        color: 'var(--text-muted)',
        marginBottom: '32px',
        letterSpacing: '1px',
      }}>
        {survived
          ? 'This passenger would likely have survived'
          : 'This passenger would likely not have survived'}
      </div>

      {/* Probability bar */}
      <div style={{ marginBottom: '8px' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '11px',
          color: 'var(--text-muted)',
          letterSpacing: '2px',
          marginBottom: '8px',
          textTransform: 'uppercase',
        }}>
          <span>Survival Probability</span>
          <span style={{ color: survived ? 'var(--accent-ice)' : 'var(--accent-danger)' }}>
            {percentage}%
          </span>
        </div>

        <div style={{
          height: '4px',
          background: 'var(--border)',
          borderRadius: '2px',
          overflow: 'hidden',
        }}>
          <div style={{
            height: '100%',
            width: `${percentage}%`,
            background: survived
              ? 'linear-gradient(90deg, #38bdf8, #7dd3fc)'
              : 'linear-gradient(90deg, #dc2626, #ef4444)',
            borderRadius: '2px',
            transition: 'width 0.8s ease',
          }} />
        </div>
      </div>

    </div>
  )
}