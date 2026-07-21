// The prebuilt cartesian dist bundle lacks carpet/contourcarpet, so we build a
// custom bundle from plotly.js/lib — these deep imports have no upstream types.
declare module 'plotly.js/lib/core' {
  import * as Plotly from 'plotly.js'
  export = Plotly
}
declare module 'plotly.js/lib/scatter' {
  const trace: unknown
  export = trace
}
declare module 'plotly.js/lib/carpet' {
  const trace: unknown
  export = trace
}
declare module 'plotly.js/lib/contourcarpet' {
  const trace: unknown
  export = trace
}
