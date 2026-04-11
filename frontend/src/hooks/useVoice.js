import { useState, useRef, useCallback } from 'react'

export function useVoice(onResult) {
  const [listening, setListening] = useState(false)
  const recog = useRef(null)

  const start = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) {
      alert('Voice input is not supported in this browser. Use Chrome or Edge.')
      return
    }

    recog.current = new SR()
    recog.current.lang = 'en-US'   // change to 'ru-RU' for Russian
    recog.current.interimResults = false
    recog.current.maxAlternatives = 1

    recog.current.onstart = () => setListening(true)
    recog.current.onend = () => setListening(false)
    recog.current.onerror = () => setListening(false)

    recog.current.onresult = (e) => {
      const text = e.results[0][0].transcript
      onResult(text)
    }

    recog.current.start()
  }, [onResult])

  const stop = useCallback(() => {
    recog.current?.stop()
    setListening(false)
  }, [])

  const toggle = useCallback(() => {
    listening ? stop() : start()
  }, [listening, start, stop])

  return { listening, toggle }
}
