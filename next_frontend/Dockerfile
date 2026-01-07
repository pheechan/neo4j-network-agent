FROM node:18-alpine AS builder
WORKDIR /app

# Copy package files and install deps
COPY package.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

# Production image
FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=5173
ENV HOSTNAME="0.0.0.0"
# Increase Node.js HTTP timeouts for slow LLM responses
ENV UV_THREADPOOL_SIZE=16

# Copy standalone build
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 5173

# Start using the standalone server
# Environment variables for timeout configuration are set above
CMD ["node", "server.js"]
