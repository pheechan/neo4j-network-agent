// Custom Next.js server with extended timeouts for slow LLM responses
const { createServer } = require('http')
const { parse } = require('url')
const next = require('next')

const dev = process.env.NODE_ENV !== 'production'
const hostname = process.env.HOSTNAME || '0.0.0.0'
const port = parseInt(process.env.PORT || '5173', 10)

const app = next({ dev, hostname, port })
const handle = app.getRequestHandler()

app.prepare().then(() => {
  const server = createServer(async (req, res) => {
    try {
      const parsedUrl = parse(req.url, true)
      await handle(req, res, parsedUrl)
    } catch (err) {
      console.error('Error occurred handling', req.url, err)
      res.statusCode = 500
      res.end('internal server error')
    }
  })

  // Set very long timeouts for LLM processing (10 minutes)
  server.timeout = 600000           // 10 minutes - total request timeout
  server.headersTimeout = 605000    // Slightly longer than timeout
  server.keepAliveTimeout = 610000  // Keep connection alive
  server.requestTimeout = 600000    // Request timeout

  server.listen(port, hostname, (err) => {
    if (err) throw err
    console.log(`> Ready on http://${hostname}:${port}`)
    console.log(`> Server timeouts configured: ${server.timeout}ms`)
  })
})
