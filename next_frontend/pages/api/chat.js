// Next.js API route - proxies chat requests to the FastAPI backend
// Uses native Node.js http module to avoid undici timeout issues

const http = require('http')

export const config = {
  api: {
    bodyParser: true,
    responseLimit: false,
    externalResolver: true, // Allow external handling
  },
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' })
  }

  try {
    // In Docker, backend is at http://backend:8000
    const backendUrl = process.env.BACKEND_URL || 'http://backend:8000'
    const url = new URL('/api/chat', backendUrl)
    
    const postData = JSON.stringify(req.body)
    
    const options = {
      hostname: url.hostname,
      port: url.port || 8000,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': Buffer.byteLength(postData),
      },
      timeout: 600000, // 10 minutes
    }

    const proxyReq = http.request(options, (proxyRes) => {
      let data = ''
      
      proxyRes.on('data', (chunk) => {
        data += chunk
      })
      
      proxyRes.on('end', () => {
        try {
          const jsonData = JSON.parse(data)
          res.status(proxyRes.statusCode).json(jsonData)
        } catch (e) {
          res.status(500).json({ detail: 'Invalid response from backend' })
        }
      })
    })

    proxyReq.on('error', (error) => {
      console.error('Proxy error:', error)
      res.status(500).json({ detail: `Failed to connect to backend: ${error.message}` })
    })

    proxyReq.on('timeout', () => {
      proxyReq.destroy()
      res.status(504).json({ detail: 'Request timed out. The LLM is taking too long. Please try again.' })
    })

    proxyReq.write(postData)
    proxyReq.end()
    
  } catch (error) {
    console.error('Proxy error:', error)
    res.status(500).json({ detail: `Failed to connect to backend: ${error.message}` })
  }
}
