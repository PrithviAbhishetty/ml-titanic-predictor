import { useState } from 'react'
import type { PassengerInput, PredictionOutput } from '../types/passenger'
import { predictSurvival } from '../api/predict'

interface Props {
  onResult: (result: PredictionOutput) => void
  onLoading: (loading: boolean) => void
  onError: (error: string | null) => void
}

const inputStyle = {
  width: '100%',
  padding: '10px 14px',
  background: '#0a0e1a',
  border: '1px solid #1e2d45',
  borderRadius: '6px',
  color: '#f0f4ff',
  fontFamily: 'DM Mono, monospace',
  fontSize: '13px',
  outline: 'none',
}

const labelStyle = {
  display: 'block' as const,
  fontSize: '11px',
  letterSpacing: '2px',
  color: '#6b7a99',
  textTransform: 'uppercase' as const,
  marginBottom: '6px',
}

const fieldStyle = {
  display: 'flex',
  flexDirection: 'column' as const,
}

export default function PredictionForm({ onResult, onLoading, onError }: Props) {
  const [form, setForm] = useState<PassengerInput>({
    pclass: 1,
    sex: 'female',
    age: 29,
    sibsp: 0,
    parch: 0,
    fare: 100,
    embarked: 'S',
  })

  function handleChange(e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    const { name, value } = e.target
    setForm(prev => ({
      ...prev,
      [name]: ['pclass', 'sibsp', 'parch'].includes(name)
        ? parseInt(value)
        : name === 'age' || name === 'fare'
        ? parseFloat(value)
        : value
    }))
  }

  async function handleSubmit() {
    onError(null)
    onLoading(true)
    try {
      const result = await predictSurvival(form)
      onResult(result)
    } catch (err) {
      onError('Failed to connect to the prediction API. Make sure the backend is running.')
    } finally {
      onLoading(false)
    }
  }

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: '12px',
      padding: '32px',
    }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '20px',
        marginBottom: '20px'
      }}>

        <div style={fieldStyle}>
          <label style={labelStyle}>Passenger Class</label>
          <select name="pclass" value={form.pclass} onChange={handleChange} style={inputStyle}>
            <option value={1}>1st Class</option>
            <option value={2}>2nd Class</option>
            <option value={3}>3rd Class</option>
          </select>
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Sex</label>
          <select name="sex" value={form.sex} onChange={handleChange} style={inputStyle}>
            <option value="female">Female</option>
            <option value="male">Male</option>
          </select>
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Age</label>
          <input
            type="number"
            name="age"
            value={form.age}
            onChange={handleChange}
            min={1}
            max={120}
            style={inputStyle}
          />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Fare (£)</label>
          <input
            type="number"
            name="fare"
            value={form.fare}
            onChange={handleChange}
            min={0}
            style={inputStyle}
          />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Siblings / Spouses Aboard</label>
          <input
            type="number"
            name="sibsp"
            value={form.sibsp}
            onChange={handleChange}
            min={0}
            style={inputStyle}
          />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Parents / Children Aboard</label>
          <input
            type="number"
            name="parch"
            value={form.parch}
            onChange={handleChange}
            min={0}
            style={inputStyle}
          />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Port of Embarkation</label>
          <select name="embarked" value={form.embarked} onChange={handleChange} style={inputStyle}>
            <option value="S">Southampton</option>
            <option value="C">Cherbourg</option>
            <option value="Q">Queenstown</option>
          </select>
        </div>

      </div>

      <button
        onClick={handleSubmit}
        style={{
          width: '100%',
          padding: '14px',
          background: 'transparent',
          border: '1px solid var(--accent-ice)',
          borderRadius: '6px',
          color: 'var(--accent-ice)',
          fontFamily: 'DM Mono, monospace',
          fontSize: '12px',
          letterSpacing: '3px',
          textTransform: 'uppercase',
          cursor: 'pointer',
          marginTop: '8px',
        }}
      >
        Predict Survival
      </button>
    </div>
  )
}