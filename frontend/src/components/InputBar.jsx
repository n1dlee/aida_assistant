import { useState, useCallback } from 'react'
import { useVoice } from '../hooks/useVoice'

export default function InputBar({ onSend, disabled }) {
  const [text, setText] = useState('')

  const handleSend = useCallback(() => {
    if (!text.trim() || disabled) return
    onSend(text)
    setText('')
  }, [text, disabled, onSend])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const { listening, toggle: toggleVoice } = useVoice((transcript) => {
    onSend(transcript)
  })

  const micColor = listening ? '#E24B4A' : 'rgba(255,255,255,0.35)'
  const micBg = listening ? 'rgba(226,75,74,0.15)' : 'transparent'

  return (
    <div style={{
      padding: '12px 16px',
      borderTop: '1px solid rgba(255,255,255,0.07)',
      display: 'flex', gap: 8, alignItems: 'flex-end',
    }}>
      {/* Voice button */}
      <button
        onClick={toggleVoice}
        title={listening ? 'Stop listening' : 'Start voice input'}
        style={{
          width: 40, height: 40, borderRadius: '50%', border: 'none',
          background: micBg, color: micColor, cursor: 'pointer',
          fontSize: 18, flexShrink: 0, transition: 'all .2s',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        {listening ? '◼' : '🎤'}
      </button>

      {/* Text input */}
      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={handleKey}
        placeholder={listening ? 'Listening...' : 'Message AIDA...'}
        disabled={disabled}
        rows={1}
        style={{
          flex: 1, resize: 'none', border: 'none', outline: 'none',
          background: 'rgba(255,255,255,0.07)',
          color: '#fff', borderRadius: 12, padding: '10px 14px',
          fontSize: 14, lineHeight: 1.5, fontFamily: 'inherit',
          maxHeight: 120, overflowY: 'auto',
        }}
      />

      {/* Send button */}
      <button
        onClick={handleSend}
        disabled={!text.trim() || disabled}
        style={{
          width: 40, height: 40, borderRadius: '50%', border: 'none',
          background: text.trim() && !disabled ? '#534AB7' : 'rgba(255,255,255,0.07)',
          color: text.trim() && !disabled ? '#fff' : 'rgba(255,255,255,0.2)',
          cursor: text.trim() && !disabled ? 'pointer' : 'default',
          fontSize: 16, flexShrink: 0, transition: 'all .2s',
        }}
      >
        ➤
      </button>
    </div>
  )
}
