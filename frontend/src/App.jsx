import { useAIDA } from './hooks/useAIDA'
import StatusBar from './components/StatusBar'
import MessageList from './components/MessageList'
import InputBar from './components/InputBar'

export default function App() {
  const { messages, status, sysInfo, send, clearHistory } = useAIDA()

  return (
    <div style={{
      height: '100dvh', display: 'flex', flexDirection: 'column',
      background: '#0f0f13', color: '#fff', fontFamily: 'system-ui, sans-serif',
      maxWidth: 720, margin: '0 auto',
    }}>
      {/* Header */}
      <div style={{
        padding: '14px 16px 10px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: '50%',
            background: 'linear-gradient(135deg, #534AB7 0%, #1D9E75 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 700, fontSize: 13, color: '#fff',
          }}>A</div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 15, letterSpacing: 1 }}>AIDA</div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', letterSpacing: 0.5 }}>
              AI Desktop Assistant
            </div>
          </div>
        </div>
        <button
          onClick={clearHistory}
          title="Clear conversation"
          style={{
            background: 'none', border: '1px solid rgba(255,255,255,0.12)',
            color: 'rgba(255,255,255,0.4)', borderRadius: 8,
            padding: '5px 10px', fontSize: 11, cursor: 'pointer',
          }}
        >
          Clear
        </button>
      </div>

      <StatusBar status={status} sysInfo={sysInfo} />
      <MessageList messages={messages} status={status} />
      <InputBar onSend={send} disabled={status === 'thinking' || status === 'connecting'} />
    </div>
  )
}
