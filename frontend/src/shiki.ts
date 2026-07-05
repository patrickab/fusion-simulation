import { createHighlighterCore, type HighlighterCore } from 'shiki/core'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'
import andromeeda from '@shikijs/themes/andromeeda'
import json from '@shikijs/langs/json'

// ponytail: trimmed copy of gigachad-bot's lib/markdown-syntax-highlighting.ts — one theme, one
// language, since this app only ever highlights JSON config blobs in a dark-only UI
let highlighterPromise: Promise<HighlighterCore> | null = null
function getHighlighter() {
  highlighterPromise ??= createHighlighterCore({
    themes: [andromeeda],
    langs: [json],
    engine: createJavaScriptRegexEngine(),
  })
  return highlighterPromise
}

// ponytail: hand-rolled, not JSON.stringify — numbers must render as scientific notation
// (2 decimals), which requires emitting them unquoted ourselves
function stringifySci(value: unknown, depth = 0): string {
  const pad = '  '.repeat(depth + 1)
  const closePad = '  '.repeat(depth)
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'number') return Number.isFinite(value) ? value.toExponential(2) : 'null'
  if (typeof value === 'boolean') return String(value)
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    const items = value.map((v) => pad + stringifySci(v, depth + 1))
    return `[\n${items.join(',\n')}\n${closePad}]`
  }
  const entries = Object.entries(value as Record<string, unknown>)
  if (entries.length === 0) return '{}'
  const items = entries.map(([k, v]) => `${pad}${JSON.stringify(k)}: ${stringifySci(v, depth + 1)}`)
  return `{\n${items.join(',\n')}\n${closePad}}`
}

export async function highlightJson(value: unknown): Promise<string> {
  const hl = await getHighlighter()
  return hl.codeToHtml(stringifySci(value), { lang: 'json', theme: 'andromeeda' })
}
