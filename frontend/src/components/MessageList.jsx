import { useEffect, useRef } from 'react'

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12,
    }}>
      {!isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: 'linear-gradient(135deg, #534AB7, #1D9E75)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, fontWeight: 600, color: '#fff',
          marginRight: 8, flexShrink: 0, marginTop: 2,
        }}>A</div>
      )}
      <div style={{
        maxWidth: '72%',
        padding: '10px 14px',
        borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
        background: isUser ? '#534AB7' : 'rgba(255,255,255,0.07)',
        color: '#fff',
        fontSize: 14,
        lineHeight: 1.6,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}>
        {msg.text}
      </div>
    </div>
  )
}

function ThinkingBubble() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
      <div style={{
        width: 28, height: 28, borderRadius: '50%',
        background: 'linear-gradient(135deg, #534AB7, #1D9E75)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 600, color: '#fff', flexShrink: 0,
      }}>A</div>
      <div style={{
        padding: '10px 16px',
        borderRadius: '16px 16px 16px 4px',
        background: 'rgba(255,255,255,0.07)',
        display: 'flex', gap: 5, alignItems: 'center',
      }}>
        {[0, 0.2, 0.4].map((delay, i) => (
          <div key={i} style={{
            width: 6, height: 6, borderRadius: '50%',
            background: '#7F77DD',
            animation: `bounce 1.2s ${delay}s ease-in-out infinite`,
          }} />
        ))}
      </div>
    </div>
  )
}

export default function MessageList({ messages, status }) {
  const bottom = useRef(null)
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, status])

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '20px 16px 8px' }}>
      {messages.length === 0 && (
        <div style={{
          height: '100%', display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          color: 'rgba(255,255,255,0.2)', fontSize: 14, gap: 8,
        }}>
          <div style={{ fontSize: 40 }}>◈</div>
          <div>Say something or type a message</div>
        </div>
      )}
      {messages.map((m, i) => <Message key={i} msg={m} />)}
      {status === 'thinking' && <ThinkingBubble />}
      <div ref={bottom} />
    </div>
  )
}
