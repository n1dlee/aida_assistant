import { useEffect, useRef, useState, useCallback } from 'react'

const WS_URL = `ws://${window.location.hostname}:8000/ws`

export function useAIDA() {
  const ws = useRef(null)
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState('connecting') // connecting | ready | thinking | error
  const [sysInfo, setSysInfo] = useState(null)

  const connect = useCallback(() => {
    ws.current = new WebSocket(WS_URL)

    ws.current.onopen = () => {
      setStatus('ready')
      fetch('/api/status').then(r => r.json()).then(setSysInfo).catch(() => {})
    }

    ws.current.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'thinking') {
        setStatus('thinking')
      } else if (msg.type === 'response') {
        setStatus('ready')
        setMessages(prev => [...prev, { role: 'assistant', text: msg.text, ts: Date.now() }])
      }
    }

    ws.current.onerror = () => setStatus('error')
    ws.current.onclose = () => {
      setStatus('connecting')
      setTimeout(connect, 2000) // auto-reconnect
    }
  }, [])

  useEffect(() => {
    connect()
    return () => ws.current?.close()
  }, [connect])

  const send = useCallback((text) => {
    if (!text.trim() || status === 'thinking') return
    setMessages(prev => [...prev, { role: 'user', text, ts: Date.now() }])
    ws.current?.send(JSON.stringify({ text }))
  }, [status])

  const clearHistory = useCallback(async () => {
    await fetch('/api/history', { method: 'DELETE' })
    setMessages([])
  }, [])

  return { messages, status, sysInfo, send, clearHistory }
}
