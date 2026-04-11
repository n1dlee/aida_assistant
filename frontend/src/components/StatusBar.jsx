export default function StatusBar({ status, sysInfo }) {
  const dots = { connecting: '●○○', ready: '●●●', thinking: '●●○', error: '○○○' }
  const colors = { connecting: '#EF9F27', ready: '#1D9E75', thinking: '#7F77DD', error: '#E24B4A' }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '8px 16px', fontSize: 12,
      borderBottom: '1px solid rgba(255,255,255,0.07)',
      color: 'rgba(255,255,255,0.5)',
    }}>
      <span style={{ color: colors[status], fontFamily: 'monospace', letterSpacing: 2 }}>
        {dots[status]}
      </span>
      <span style={{ color: colors[status], textTransform: 'uppercase', letterSpacing: 1 }}>
        {status}
      </span>
      {sysInfo && (
        <>
          <span style={{ marginLeft: 'auto' }}>
            {sysInfo.local_llm ? '⬡ local' : ''}
            {sysInfo.cloud_llm ? '  ☁ cloud' : ''}
          </span>
          <span>{sysInfo.memory_entries} memories</span>
        </>
      )}
    </div>
  )
}
