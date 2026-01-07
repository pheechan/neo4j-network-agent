/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  // Increase timeouts for LLM queries which can take a long time
  experimental: {
    proxyTimeout: 600000, // 10 minutes
  },
  // Server-side runtime configuration
  serverRuntimeConfig: {
    // Long timeout for API routes
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
}

module.exports = nextConfig
